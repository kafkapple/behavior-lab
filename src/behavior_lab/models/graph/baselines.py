"""Baseline graph models: ST-GCN and AGCN."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import bn_init, conv_init
from behavior_lab.core.graph import Graph


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, partitions=3):
        super().__init__()
        self.num_partitions = partitions
        self.A = nn.Parameter(
            torch.from_numpy(A.astype(np.float32)).unsqueeze(0).repeat(partitions, 1, 1),
            requires_grad=True)
        self.conv = nn.Conv2d(in_channels, out_channels * partitions, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels * partitions)
        self.out_c = out_channels
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        N, C, T, V = x.size()
        x = self.bn(self.conv(x))
        x = x.view(N, self.num_partitions, self.out_c, T, V)
        return torch.einsum('nkctv,kvw->nctw', x, self.A)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels))
        else:
            self.residual = lambda x: x

    def forward(self, x):
        return self.relu(self.tcn(self.relu(self.gcn(x))) + self.residual(x))


class STGCN(nn.Module):
    """Spatial Temporal Graph Convolutional Network (Yan et al., AAAI 2018)."""

    def __init__(self, num_classes, num_joints, num_persons=2, in_channels=3,
                 skeleton='ntu', **kwargs):
        super().__init__()
        graph = Graph(skeleton)
        A = graph.A_norm
        self.data_bn = nn.BatchNorm1d(num_persons * in_channels * num_joints)
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A), STGCNBlock(64, 64, A), STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2), STGCNBlock(128, 128, A), STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2), STGCNBlock(256, 256, A), STGCNBlock(256, 256, A)])
        self.fc = nn.Linear(256, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))
        bn_init(self.data_bn, 1)
        self.num_persons = num_persons

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for layer in self.layers:
            x = layer(x)
        x = x.view(N, M, -1, x.size(2), V).mean(4).mean(1)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc(x)


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.PA = nn.Parameter(torch.zeros_like(self.A))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_a = nn.Conv1d(in_channels, out_channels // 4, 1)
        self.conv_b = nn.Conv1d(in_channels, out_channels // 4, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A + self.PA
        x_a = self.conv_a(x.mean(2))
        x_b = self.conv_b(x.mean(2))
        adapt_A = torch.softmax(torch.bmm(x_a.transpose(1, 2), x_b), dim=-1)
        A = A.unsqueeze(0) + self.alpha * adapt_A
        x = torch.einsum('nctv,nvw->nctw', self.bn(self.conv(x)), A)
        return x


class AGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels))
        else:
            self.residual = lambda x: x

    def forward(self, x):
        return self.relu(self.tcn(self.relu(self.gcn(x))) + self.residual(x))


class AGCN(nn.Module):
    """Adaptive Graph Convolutional Network (Shi et al., CVPR 2019)."""

    def __init__(self, num_classes, num_joints, num_persons=2, in_channels=3,
                 skeleton='ntu', **kwargs):
        super().__init__()
        graph = Graph(skeleton)
        A = graph.A_norm
        self.data_bn = nn.BatchNorm1d(num_persons * in_channels * num_joints)
        self.layers = nn.ModuleList([
            AGCNBlock(in_channels, 64, A), AGCNBlock(64, 64, A), AGCNBlock(64, 64, A),
            AGCNBlock(64, 128, A, stride=2), AGCNBlock(128, 128, A), AGCNBlock(128, 128, A),
            AGCNBlock(128, 256, A, stride=2), AGCNBlock(256, 256, A), AGCNBlock(256, 256, A)])
        self.fc = nn.Linear(256, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))
        bn_init(self.data_bn, 1)
        self.num_persons = num_persons

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for layer in self.layers:
            x = layer(x)
        x = x.view(N, M, -1, x.size(2), V).mean(4).mean(1)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc(x)
