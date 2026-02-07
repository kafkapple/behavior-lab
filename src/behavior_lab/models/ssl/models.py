"""Unified SSL model builder combining encoders and methods."""
import torch
import torch.nn as nn
from typing import Dict

from .encoders import get_encoder
from .methods import get_ssl_method


class SSLModel(nn.Module):
    """Wrapper: encoder + SSL method with unified interface."""

    def __init__(self, encoder_type, ssl_method, in_channels=2, embed_dim=256,
                 num_joints=7, num_frames=64, num_subjects=2, **kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        self.ssl_method_type = ssl_method
        self.embed_dim = embed_dim

        enc_kwargs = dict(in_channels=in_channels, out_channels=embed_dim,
                          num_joints=num_joints, num_frames=num_frames, num_subjects=num_subjects)
        if encoder_type.lower() == 'infogcn':
            enc_kwargs['use_vae'] = ssl_method.lower() not in ['mae']
        encoder = get_encoder(encoder_type, **enc_kwargs)

        method_kwargs = dict(embed_dim=embed_dim, num_joints=num_joints,
                             num_frames=num_frames, num_subjects=num_subjects)
        method_kwargs.update(kwargs)
        self.method = get_ssl_method(ssl_method, encoder, **method_kwargs)

    def forward(self, x): return self.method(x)
    def encode(self, x): return self.method.encode(x)

    @property
    def num_parameters(self): return sum(p.numel() for p in self.parameters())


def build_ssl_model(encoder='gcn', ssl_method='dino', lite=False, **kwargs):
    """Build SSL model. Use lite=True for smaller debug models."""
    if lite:
        kwargs['embed_dim'] = kwargs.get('embed_dim', 256) // 2
    return SSLModel(encoder, ssl_method, **kwargs)


MODEL_REGISTRY = {
    'gcn_mae': lambda **kw: build_ssl_model('gcn', 'mae', **kw),
    'gcn_jepa': lambda **kw: build_ssl_model('gcn', 'jepa', **kw),
    'gcn_dino': lambda **kw: build_ssl_model('gcn', 'dino', **kw),
    'infogcn_mae': lambda **kw: build_ssl_model('infogcn', 'mae', **kw),
    'infogcn_jepa': lambda **kw: build_ssl_model('infogcn', 'jepa', **kw),
    'infogcn_dino': lambda **kw: build_ssl_model('infogcn', 'dino', **kw),
    'interaction_mae': lambda **kw: build_ssl_model('interaction', 'mae', **kw),
    'interaction_jepa': lambda **kw: build_ssl_model('interaction', 'jepa', **kw),
    'interaction_dino': lambda **kw: build_ssl_model('interaction', 'dino', **kw),
}
