"""SSL Encoders: GCN, InfoGCN, InfoGCN-Interaction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from einops import rearrange


class GraphConvolution(nn.Module):
    """Basic graph convolution layer with learnable adjacency."""
    def __init__(self, in_channels, out_channels, num_joints=7,
                 temporal_kernel=9, stride=1, residual=True, dropout=0.0):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(num_joints, num_joints))
        nn.init.uniform_(self.A, -0.01, 0.01)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        pad = (temporal_kernel - 1) // 2
        self.temporal_conv = nn.Conv2d(out_channels, out_channels,
            kernel_size=(temporal_kernel, 1), stride=(stride, 1), padding=(pad, 0), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        N, C, T, V = x.shape
        A = F.softmax(self.A[:V, :V] + torch.eye(V, device=x.device), dim=-1)
        res = self.residual(x)
        x_flat = x.permute(0, 2, 1, 3).reshape(N * T, C, V)
        x_agg = torch.matmul(x_flat, A).reshape(N, T, C, V).permute(0, 2, 1, 3)
        x = self.dropout(self.bn(self.temporal_conv(self.spatial_conv(x_agg))))
        return self.relu(x + res)


class GCNEncoder(nn.Module):
    """Simple GCN encoder. Input: (N,C,T,V,M) -> Output: (N, embed_dim)."""
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=256,
                 num_joints=7, num_frames=64, num_subjects=2, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_subjects = num_subjects
        self.out_channels = out_channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        channels = [hidden_channels] + [hidden_channels * (2 ** min(i, 2)) for i in range(num_layers - 1)]
        channels.append(out_channels)
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(channels[i], channels[i + 1], num_joints,
                             stride=2 if i == num_layers // 2 else 1, dropout=dropout)
            for i in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, mask=None, return_tokens=False):
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)
        if mask is not None:
            m = mask.unsqueeze(1).expand(-1, C, -1, -1, -1)
            m = m.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)
            x = x * (1 - m.float())
        x = self.input_proj(x)
        for layer in self.gcn_layers:
            x = layer(x)
        if return_tokens:
            return x.reshape(N, M, -1, x.shape[2], x.shape[3]).mean(dim=1)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return x.reshape(N, M, -1).mean(dim=1)


class InfoGCNEncoder(nn.Module):
    """InfoGCN-style encoder with attention and optional VAE.
    Input: (N,C,T,V,M) -> Output: (N, embed_dim) or (z, mu, logvar)."""
    def __init__(self, in_channels=2, base_channel=64, out_channels=256,
                 num_joints=7, num_frames=64, num_subjects=2, num_head=3,
                 dropout=0.0, use_vae=True):
        super().__init__()
        self.num_joints = num_joints
        self.num_subjects = num_subjects
        self.out_channels = out_channels
        self.use_vae = use_vae

        self.data_bn = nn.BatchNorm1d(num_subjects * base_channel * num_joints)
        self.A = nn.Parameter(self._build_adj(num_joints))
        self.joint_embed = nn.Linear(in_channels, base_channel)
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, base_channel))
        self.layers = nn.ModuleList([
            self._make_layer(base_channel, base_channel, stride=1),
            self._make_layer(base_channel, base_channel, stride=1),
            self._make_layer(base_channel, base_channel * 2, stride=2),
            self._make_layer(base_channel * 2, base_channel * 2, stride=1),
            self._make_layer(base_channel * 2, base_channel * 4, stride=2),
            self._make_layer(base_channel * 4, base_channel * 4, stride=1)])
        self.fc = nn.Linear(base_channel * 4, out_channels)
        if use_vae:
            self.fc_mu = nn.Linear(out_channels, out_channels)
            self.fc_logvar = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _build_adj(self, V):
        A = torch.zeros(V, V)
        edges = [(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
        for i, j in edges:
            if i < V and j < V: A[i, j] = A[j, i] = 1
        A += torch.eye(V)
        return A / A.sum(dim=1, keepdim=True).clamp(min=1)

    def _make_layer(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, mask=None, return_tokens=False):
        N, C, T, V, M = x.shape
        x = rearrange(x.contiguous(), 'n c t v m -> (n m t) v c').contiguous()
        A = F.softmax(self.A.to(x.device), dim=-1)
        x = torch.matmul(A.expand(x.shape[0], -1, -1), x)
        x = self.joint_embed(x) + self.pos_embed
        x = rearrange(x, '(n m t) v c -> (n m) c t v', n=N, m=M, t=T).contiguous()
        if mask is not None:
            mm = mask.unsqueeze(1).repeat(1, x.shape[1], 1, 1, 1)
            mm = rearrange(mm, 'n c t v m -> (n m) c t v')
            x = x * (1 - mm.float())
        for layer in self.layers:
            x = layer(x)
        if return_tokens:
            return rearrange(x, '(n m) c t v -> n m c t v', n=N, m=M).mean(dim=1)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = rearrange(x, '(n m) c -> n m c', n=N, m=M).mean(dim=1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        if self.use_vae:
            mu, logvar = self.fc_mu(x), self.fc_logvar(x)
            if self.training:
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            else:
                z = mu
            return z, mu, logvar
        return x


class InfoGCNInteractionEncoder(InfoGCNEncoder):
    """InfoGCN with cross-attention + interaction pooling for multi-subject."""
    def __init__(self, interaction_type='full', **kwargs):
        super().__init__(**kwargs)
        ch = kwargs.get('base_channel', 64) * 4
        M = kwargs.get('num_subjects', 2)
        self.use_cross_attn = interaction_type in ['attention', 'full'] and M > 1
        self.use_interaction_pool = interaction_type in ['pooling', 'full'] and M > 1
        if self.use_cross_attn:
            self.subject_attn = nn.MultiheadAttention(ch, 4, dropout=kwargs.get('dropout', 0.0), batch_first=True)
            self.attn_norm = nn.LayerNorm(ch)
        if self.use_interaction_pool:
            self.interaction_proj = nn.Sequential(
                nn.Linear(ch * 4, ch * 2), nn.ReLU(inplace=True), nn.Linear(ch * 2, ch))

    def forward(self, x, mask=None, return_tokens=False):
        N, C, T, V, M = x.shape
        x = rearrange(x.contiguous(), 'n c t v m -> (n m t) v c').contiguous()
        A = F.softmax(self.A[:V, :V], dim=-1)
        x = torch.matmul(A.to(x.device), x)
        x = self.joint_embed(x) + self.pos_embed[:, :V]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', n=N, m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        for layer in self.layers:
            x = layer(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x_p = x.mean(dim=-1)
        if M > 1:
            if self.use_cross_attn:
                xn = self.attn_norm(x_p)
                ao, _ = self.subject_attn(xn, xn, xn)
                x_p = x_p + ao
            if self.use_interaction_pool:
                f1, f2 = x_p[:, 0], x_p[:, 1]
                x = self.interaction_proj(torch.cat([f1, f2, f1 * f2, torch.abs(f1 - f2)], dim=-1))
            else:
                x = x_p.mean(dim=1)
        else:
            x = x_p.squeeze(1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        if self.use_vae:
            mu, logvar = self.fc_mu(x), self.fc_logvar(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            return z, mu, logvar
        return x


def get_encoder(encoder_type, **kwargs):
    """Factory: get encoder by name."""
    t = encoder_type.lower()
    if t == 'gcn': return GCNEncoder(**kwargs)
    elif t == 'infogcn': return InfoGCNEncoder(**kwargs)
    elif t in ('infogcn_interaction', 'interaction'):
        return InfoGCNInteractionEncoder(interaction_type=kwargs.pop('interaction_type', 'full'), **kwargs)
    raise ValueError(f"Unknown encoder: {encoder_type}")
