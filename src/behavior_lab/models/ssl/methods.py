"""SSL Methods: MAE, JEPA, DINO."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, Tuple

from behavior_lab.data.preprocessing.augmentation import SkeletonAugmentor


class MAEMethod(nn.Module):
    """Masked Autoencoder: reconstruct masked joints in pixel space."""
    def __init__(self, encoder, in_channels=2, embed_dim=256, num_joints=7,
                 num_frames=64, num_subjects=2, mask_ratio=0.4, decoder_depth=2):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        out_size = in_channels * num_joints * num_frames * num_subjects
        layers = []
        for i in range(decoder_depth):
            in_d = embed_dim if i == 0 else embed_dim // 2
            out_d = embed_dim // 2 if i < decoder_depth - 1 else out_size
            layers.extend([nn.Linear(in_d, out_d), nn.ReLU() if i < decoder_depth - 1 else nn.Identity()])
        self.decoder = nn.Sequential(*layers)
        self.projection = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.in_channels = in_channels

    def generate_mask(self, x):
        N, C, T, V, M = x.shape
        mask = torch.zeros(N, T, V, M, device=x.device)
        num_mask = int(T * self.mask_ratio)
        for i in range(N):
            mask[i, torch.randperm(T, device=x.device)[:num_mask]] = 1
        return mask

    def forward(self, x):
        N, C, T, V, M = x.shape
        mask = self.generate_mask(x)
        me = mask.unsqueeze(1).expand(-1, C, -1, -1, -1)
        z = self.encoder(x * (1 - me), mask=mask, return_tokens=False)
        if isinstance(z, tuple): z = z[0]
        recon = self.decoder(z)
        expected = N * C * T * V * M
        if recon.numel() != expected:
            recon = nn.Linear(recon.view(N, -1).shape[1], C * T * V * M, device=x.device)(recon.view(N, -1))
        recon = recon.reshape(N, C, T, V, M)
        diff = ((recon - x) ** 2) * me
        loss = diff.sum() / (me.sum() + 1e-8)
        return {'loss': loss, 'z': z, 'projection': self.projection(z), 'mask': mask}

    def encode(self, x):
        with torch.no_grad():
            z = self.encoder(x, mask=None, return_tokens=False)
            return z[0] if isinstance(z, tuple) else z


class JEPAMethod(nn.Module):
    """Joint-Embedding Predictive Architecture: predict target in latent space."""
    def __init__(self, encoder, embed_dim=256, predictor_dim=128, num_joints=7,
                 num_frames=64, num_subjects=2, mask_ratio=0.4, ema_momentum=0.996):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ema_momentum = ema_momentum
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim), nn.GELU(),
            nn.Linear(predictor_dim, predictor_dim), nn.GELU(),
            nn.Linear(predictor_dim, embed_dim))
        self.projection = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def generate_mask(self, x):
        N, C, T, V, M = x.shape
        mask = torch.zeros(N, T, V, M, device=x.device)
        n = int(T * self.mask_ratio)
        for i in range(N):
            mask[i, torch.randperm(T, device=x.device)[:n]] = 1
        return mask

    @torch.no_grad()
    def update_target(self):
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.data = self.ema_momentum * pt.data + (1 - self.ema_momentum) * pc.data

    def forward(self, x):
        mask = self.generate_mask(x)
        zc = self.context_encoder(x, mask=mask, return_tokens=False)
        if isinstance(zc, tuple): zc = zc[0]
        zp = self.predictor(zc)
        with torch.no_grad():
            zt = self.target_encoder(x, mask=None, return_tokens=False)
            if isinstance(zt, tuple): zt = zt[0]
        loss = F.mse_loss(zp, zt.detach())
        if self.training: self.update_target()
        return {'loss': loss, 'z': zc, 'projection': self.projection(zc), 'mask': mask}

    def encode(self, x):
        with torch.no_grad():
            z = self.context_encoder(x, mask=None, return_tokens=False)
            return z[0] if isinstance(z, tuple) else z


class DINOMethod(nn.Module):
    """Self-Distillation with No Labels: teacher-student with centering."""
    def __init__(self, encoder, embed_dim=256, out_dim=256, num_joints=7,
                 num_frames=64, num_subjects=2, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9, ema_momentum=0.996):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ema_momentum = ema_momentum
        self.student_encoder = encoder
        self.student_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2), nn.GELU(),
            nn.Linear(embed_dim * 2, out_dim))
        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher_encoder.parameters(): p.requires_grad = False
        for p in self.teacher_head.parameters(): p.requires_grad = False
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.strong_aug = SkeletonAugmentor(mode='strong')
        self.weak_aug = SkeletonAugmentor(mode='weak')

    @torch.no_grad()
    def update_teacher(self):
        for pt, ps in zip(self.teacher_encoder.parameters(), self.student_encoder.parameters()):
            pt.data = self.ema_momentum * pt.data + (1 - self.ema_momentum) * ps.data
        for pt, ps in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            pt.data = self.ema_momentum * pt.data + (1 - self.ema_momentum) * ps.data

    def _encode(self, encoder, x):
        out = encoder(x)
        return out[0] if isinstance(out, tuple) else out

    def forward(self, x):
        dev = x.device
        xs = self.strong_aug(x).float().to(dev)
        xw = self.weak_aug(x).float().to(dev)
        sf = self._encode(self.student_encoder, xs)
        so = self.student_head(sf)
        with torch.no_grad():
            tf = self._encode(self.teacher_encoder, xw)
            to_ = self.teacher_head(tf)
        tc = to_ - self.center
        tp = F.softmax(tc / self.teacher_temp, dim=-1)
        slp = F.log_softmax(so / self.student_temp, dim=-1)
        loss = -torch.sum(tp * slp, dim=-1).mean()
        if self.training:
            self.center = self.center_momentum * self.center + (1 - self.center_momentum) * to_.mean(0, keepdim=True)
            self.update_teacher()
        return {'loss': loss, 'projection': sf}

    def encode(self, x):
        with torch.no_grad():
            return self._encode(self.student_encoder, x)


def get_ssl_method(method_type, encoder, embed_dim=256, **kwargs):
    """Factory: get SSL method by name."""
    t = method_type.lower()
    if t == 'mae': return MAEMethod(encoder, embed_dim=embed_dim, **kwargs)
    elif t == 'jepa': return JEPAMethod(encoder, embed_dim=embed_dim, **kwargs)
    elif t == 'dino': return DINOMethod(encoder, embed_dim=embed_dim, **kwargs)
    raise ValueError(f"Unknown SSL method: {method_type}")
