"""Model registry: unified access to all models."""
from .graph_models import InfoGCN, InfoGCN_Interaction, STGCN, AGCN
from .sequence_models import get_action_classifier
from .ssl import build_ssl_model, MODEL_REGISTRY as SSL_REGISTRY


def _remap_infogcn_kwargs(kwargs: dict) -> dict:
    """Map unified names to InfoGCN-specific parameter names."""
    mapping = {'num_classes': 'num_class', 'num_joints': 'num_point',
               'num_persons': 'num_person', 'num_channels': 'in_channels'}
    kw = {mapping.get(k, k): v for k, v in kwargs.items()}
    kw.pop('A', None)  # InfoGCN builds its own adjacency from skeleton name
    return kw


def get_model(name: str, **kwargs):
    """Get any model by name.

    Graph models: 'infogcn', 'stgcn', 'agcn'
    Sequence models: 'lstm', 'mlp', 'transformer', 'rule_based'
    SSL models: 'gcn_dino', 'infogcn_jepa', 'interaction_mae', etc.
    """
    name_lower = name.lower()

    # SSL models
    if name_lower in SSL_REGISTRY:
        return SSL_REGISTRY[name_lower](**kwargs)

    # Graph models
    graph_models = {'infogcn': InfoGCN, 'infogcn_interaction': InfoGCN_Interaction,
                    'stgcn': STGCN, 'agcn': AGCN}
    if name_lower in graph_models:
        kw = _remap_infogcn_kwargs(kwargs) if 'infogcn' in name_lower else kwargs
        return graph_models[name_lower](**kw)

    # Sequence models
    seq_names = ('lstm', 'mlp', 'transformer', 'rule_based', 'rule', 'baseline')
    if name_lower in seq_names:
        return get_action_classifier(name_lower, **kwargs)

    available = list(SSL_REGISTRY.keys()) + list(graph_models.keys()) + list(seq_names)
    raise ValueError(f"Unknown model: {name}. Available: {available}")
