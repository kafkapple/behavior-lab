"""InfoGCN: Information Bottleneck GCN for skeleton action recognition."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .modules import EncodingBlock, InteractionPooling, SubjectCrossAttention, bn_init
from behavior_lab.core.graph import Graph


class InfoGCN(nn.Module):
    """InfoGCN with attention-based graph convolution and information bottleneck.
    
    Input: (N, C, T, V, M)
    Output: (logits, z) where z is the latent representation
    """

    def __init__(self, num_class: int = 60, num_point: int = 25, num_person: int = 2,
                 in_channels: int = 3, skeleton: str = 'ntu', base_channel: int = 64,
                 num_head: int = 3, k: int = 1, drop_out: float = 0,
                 noise_ratio: float = 0.1, gain: float = 3):
        super().__init__()
        A = np.stack([np.eye(num_point)] * num_head, axis=0)
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.noise_ratio = noise_ratio

        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        self.z_prior = torch.empty(num_class, base_channel * 4)
        nn.init.orthogonal_(self.z_prior, gain=gain)

        # Graph transformation from skeleton
        graph = Graph(skeleton)
        A_outward = graph.A_outward_binary
        I = np.eye(graph.num_node)
        self.A_vector = torch.from_numpy(
            (I - np.linalg.matrix_power(A_outward, k)).astype(np.float32))

        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_point, base_channel))

        # 9 encoding blocks
        self.l1 = EncodingBlock(base_channel, base_channel, A)
        self.l2 = EncodingBlock(base_channel, base_channel, A)
        self.l3 = EncodingBlock(base_channel, base_channel, A)
        self.l4 = EncodingBlock(base_channel, base_channel * 2, A, stride=2)
        self.l5 = EncodingBlock(base_channel * 2, base_channel * 2, A)
        self.l6 = EncodingBlock(base_channel * 2, base_channel * 2, A)
        self.l7 = EncodingBlock(base_channel * 2, base_channel * 4, A, stride=2)
        self.l8 = EncodingBlock(base_channel * 4, base_channel * 4, A)
        self.l9 = EncodingBlock(base_channel * 4, base_channel * 4, A)

        self.fc = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_mu = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_logvar = nn.Linear(base_channel * 4, base_channel * 4)
        self.decoder = nn.Linear(base_channel * 4, num_class)

        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N * M * T, -1, -1) @ x
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5,
                      self.l6, self.l7, self.l8, self.l9]:
            x = layer(x)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1).mean(3).mean(1)
        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        if self.training:
            std = z_logvar.mul(self.noise_ratio).exp().clamp(max=100)
            z = torch.empty_like(std).normal_() * std + z_mu
        else:
            z = z_mu

        return self.decoder(z), z


class InfoGCN_Interaction(InfoGCN):
    """InfoGCN with multi-subject interaction modeling (cross-attention + pooling)."""

    def __init__(self, interaction_type: str = 'full', **kwargs):
        super().__init__(**kwargs)
        base_channel = kwargs.get('base_channel', 64)
        drop_out = kwargs.get('drop_out', 0)
        num_person = kwargs.get('num_person', 2)

        self.use_cross_attention = interaction_type in ['attention', 'full']
        self.use_interaction_pooling = interaction_type in ['pooling', 'full']

        if self.use_cross_attention:
            self.subject_attention = SubjectCrossAttention(
                base_channel * 4, num_heads=4, dropout=drop_out or 0.1)
        if self.use_interaction_pooling:
            self.interaction_pool = InteractionPooling(base_channel * 4, num_person)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N * M * T, -1, -1) @ x
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5,
                      self.l6, self.l7, self.l8, self.l9]:
            x = layer(x)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)

        if self.use_cross_attention and M > 1:
            x_attn = x.mean(dim=-1)
            x_attn = self.subject_attention(x_attn)
            x = x + x_attn.unsqueeze(-1)

        if self.use_interaction_pooling and M > 1:
            x = self.interaction_pool(x)
        else:
            x = x.mean(dim=-1).mean(dim=1)

        x = F.relu(self.fc(x))
        x = self.drop_out(x)
        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        if self.training:
            std = z_logvar.mul(self.noise_ratio).exp().clamp(max=100)
            z = torch.empty_like(std).normal_() * std + z_mu
        else:
            z = z_mu

        return self.decoder(z), z
