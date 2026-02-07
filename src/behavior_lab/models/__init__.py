"""Model registry: unified access to all models.

Factory pattern for instantiating any model by name + Hydra config.

Taxonomy (single axis: Architecture, sub-grouped by paradigm):

- graph/       GCN family — InfoGCN, STGCN, AGCN (self) + PySKL wrappers (external)
- sequence/    Sequence classifiers — MLP, LSTM, Transformer, RuleBased
- ssl/         Self-supervised — 3 encoders × 3 methods = 9 combinations
- discovery/   Behavior discovery — B-SOiD, MoSeq, SUBTLE, BehaveMAE, clustering
- losses/      Label smoothing, MMD
"""
from .graph import InfoGCN, InfoGCN_Interaction, STGCN, AGCN
from .sequence import get_action_classifier
from .ssl import build_ssl_model, MODEL_REGISTRY as SSL_REGISTRY


def _remap_infogcn_kwargs(kwargs: dict) -> dict:
    """Map unified names to InfoGCN-specific parameter names."""
    mapping = {'num_classes': 'num_class', 'num_joints': 'num_point',
               'num_persons': 'num_person', 'num_channels': 'in_channels'}
    kw = {mapping.get(k, k): v for k, v in kwargs.items()}
    kw.pop('A', None)  # InfoGCN builds its own adjacency from skeleton name
    return kw


# Lazy-loaded discovery model constructors (heavy optional deps)
_DISCOVERY_MODELS = {
    'bsoid': ('behavior_lab.models.discovery.bsoid', 'BSOiD'),
    'b_soid': ('behavior_lab.models.discovery.bsoid', 'BSOiD'),
    'moseq': ('behavior_lab.models.discovery.moseq', 'KeypointMoSeq'),
    'keypoint_moseq': ('behavior_lab.models.discovery.moseq', 'KeypointMoSeq'),
    'subtle': ('behavior_lab.models.discovery.subtle_wrapper', 'SUBTLE'),
    'behavemae': ('behavior_lab.models.discovery.behavemae', 'BehaveMAE'),
    'behave_mae': ('behavior_lab.models.discovery.behavemae', 'BehaveMAE'),
    'clustering': ('behavior_lab.models.discovery.clustering', 'cluster_features'),
}

# Lazy-loaded PySKL graph models (mmskeleton successor)
_PYSKL_MODELS = {
    'stgcn_pyskl': 'stgcn',
    'stgcn++_pyskl': 'stgcn++',
    'ctrgcn_pyskl': 'ctrgcn',
    'aagcn_pyskl': 'aagcn',
    'msg3d_pyskl': 'msg3d',
    'dgstgcn_pyskl': 'dgstgcn',
    'poseconv3d_pyskl': 'poseconv3d',
}


def _lazy_import(module_path: str, class_name: str):
    """Import a class from a module path lazily."""
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_model(name: str, **kwargs):
    """Get any model by name.

    Graph (supervised):    'infogcn', 'stgcn', 'agcn'
    Graph (PySKL):         'stgcn_pyskl', 'ctrgcn_pyskl', 'msg3d_pyskl', ...
    Sequence (supervised): 'lstm', 'mlp', 'transformer', 'rule_based'
    SSL:                   'gcn_dino', 'infogcn_jepa', 'interaction_mae', ...
    Discovery:             'bsoid', 'moseq', 'subtle', 'behavemae'
    """
    name_lower = name.lower().replace('-', '_')

    # SSL models
    if name_lower in SSL_REGISTRY:
        return SSL_REGISTRY[name_lower](**kwargs)

    # Graph models (self-implemented)
    graph_models = {
        'infogcn': InfoGCN,
        'infogcn_interaction': InfoGCN_Interaction,
        'stgcn': STGCN,
        'agcn': AGCN,
    }
    if name_lower in graph_models:
        kw = _remap_infogcn_kwargs(kwargs) if 'infogcn' in name_lower else kwargs
        return graph_models[name_lower](**kw)

    # Graph models (PySKL external)
    if name_lower in _PYSKL_MODELS:
        PySKLModel = _lazy_import('behavior_lab.models.graph.pyskl', 'PySKLModel')
        preset_name = _PYSKL_MODELS[name_lower]
        if 'checkpoint_path' in kwargs and 'config_path' in kwargs:
            return PySKLModel.from_checkpoint(
                kwargs.pop('config_path'), kwargs.pop('checkpoint_path'),
                device=kwargs.pop('device', 'cpu'))
        return PySKLModel.from_config(model_name=preset_name, **kwargs)

    # Sequence models
    seq_names = ('lstm', 'mlp', 'transformer', 'rule_based', 'rule', 'baseline')
    if name_lower in seq_names:
        return get_action_classifier(name_lower, **kwargs)

    # Discovery models (lazy import — heavy deps)
    if name_lower in _DISCOVERY_MODELS:
        module_path, class_name = _DISCOVERY_MODELS[name_lower]
        cls = _lazy_import(module_path, class_name)
        if name_lower in ('behavemae', 'behave_mae') and 'checkpoint_path' in kwargs:
            return cls.from_pretrained(**kwargs)
        if name_lower == 'clustering':
            return cls  # Returns the function itself
        return cls(**kwargs)

    # Collect all available model names
    available = (
        list(SSL_REGISTRY.keys()) +
        list(graph_models.keys()) +
        list(_PYSKL_MODELS.keys()) +
        list(seq_names) +
        list(_DISCOVERY_MODELS.keys())
    )
    raise ValueError(f"Unknown model: {name}. Available: {sorted(set(available))}")


def list_models() -> dict:
    """List all available model names grouped by category."""
    return {
        'graph': ['infogcn', 'infogcn_interaction', 'stgcn', 'agcn'],
        'graph_pyskl': sorted(_PYSKL_MODELS.keys()),
        'sequence': ['mlp', 'lstm', 'transformer', 'rule_based'],
        'ssl': sorted(SSL_REGISTRY.keys()),
        'discovery': sorted(_DISCOVERY_MODELS.keys()),
    }
