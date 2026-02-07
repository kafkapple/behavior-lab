"""Neural network modules for GCN-based skeleton action recognition."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange


def bn_init(bn: nn.BatchNorm2d, scale: float):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_init(conv: nn.Conv2d):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_branch_init(conv: nn.Conv2d, branches: int):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class MultiScale_TemporalConv(nn.Module):
    """Multi-scale temporal convolution with dilated branches + maxpool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5,
                 stride: int = 1, dilations: list = [1, 2], residual: bool = True,
                 residual_kernel_size: int = 1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_channels, branch_channels,
                          kernel_size=(kernel_size, 1),
                          padding=((kernel_size - 1) * d // 2, 0),
                          dilation=(d, 1), stride=(stride, 1)),
                nn.BatchNorm2d(branch_channels)
            ) for d in dilations
        ])
        # Max pool branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))
        # 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return out + self.residual(x)


class SelfAttention(nn.Module):
    """Joint self-attention for skeleton graphs."""
    def __init__(self, in_channels: int, hidden_dim: int, n_heads: int):
        super().__init__()
        self.scale = hidden_dim ** -0.5
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim * 2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x):
        y = rearrange(x, 'n c t v -> n t v c').contiguous()
        y = self.ln(y)
        y = self.to_qk(y)
        qk = y.chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b t v (h d) -> (b t) h v d', h=self.n_heads), qk)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        return dots.softmax(dim=-1).float()


class SA_GC(nn.Module):
    """Self-Attention Graph Convolution with learnable topology."""
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray):
        super().__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head = A.shape[0]

        self.shared_topology = nn.Parameter(
            torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        self.conv_d = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for _ in range(self.num_head)])

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d): conv_init(m)
            elif isinstance(m, nn.BatchNorm2d): bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

        rel_channels = in_channels // 8
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)

    def forward(self, x, attn=None):
        N, C, T, V = x.size()
        if attn is None:
            attn = self.attn(x)
        A = attn * self.shared_topology.unsqueeze(0)

        out = None
        for h in range(self.num_head):
            A_h = A[:, h, :, :]
            feature = rearrange(x, 'n c t v -> (n t) v c')
            z = A_h @ feature
            z = rearrange(z, '(n t) v c -> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        return self.relu(out + self.down(x))


class EncodingBlock(nn.Module):
    """SA-GC + Multi-Scale TCN + residual."""
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray,
                 stride: int = 1, residual: bool = True):
        super().__init__()
        self.agcn = SA_GC(in_channels, out_channels, A)
        self.tcn = MultiScale_TemporalConv(
            out_channels, out_channels, kernel_size=5, stride=stride,
            dilations=[1, 2], residual=False)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1)),
                nn.BatchNorm2d(out_channels))

    def forward(self, x, attn=None):
        return self.relu(self.tcn(self.agcn(x, attn)) + self.residual(x))


class InteractionPooling(nn.Module):
    """Interaction-aware pooling: [f1, f2, f1*f2, |f1-f2|] -> projection."""
    def __init__(self, in_channels: int, num_subjects: int = 2):
        super().__init__()
        self.num_subjects = num_subjects
        self.projection = nn.Sequential(
            nn.Linear((num_subjects + 2) * in_channels, in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 2, in_channels))

    def forward(self, x):
        """x: (N, M, C, ...) or (N, M, C)"""
        if x.dim() == 4:
            x = x.mean(dim=-1)
        N, M, C = x.shape[:3]
        if M == 1:
            return x[:, 0]

        features = [x[:, m] for m in range(M)]
        interaction = features[0] * features[1]
        difference = torch.abs(features[0] - features[1])
        combined = torch.cat(features + [interaction, difference], dim=-1)
        return self.projection(combined)


class SubjectCrossAttention(nn.Module):
    """Cross-attention between subjects for interaction modeling."""
    def __init__(self, in_channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels), nn.Dropout(dropout))

    def forward(self, x):
        """x: (N, M, C)"""
        N, M, C = x.shape
        if M == 1:
            return x
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(N, M, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(N, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(N, M, self.num_heads, self.head_dim).transpose(1, 2)
        attn = self.dropout(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(N, M, C)
        x = x + self.out_proj(out)
        return x + self.ffn(self.norm2(x))
