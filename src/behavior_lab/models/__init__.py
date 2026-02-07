"""Model registry: unified access to all models.

Factory pattern for instantiating any model by name + Hydra config.
Models are grouped into categories:

- Graph models (supervised): infogcn, stgcn, agcn
- Sequence models (supervised): mlp, lstm, transformer, rule_based
- SSL models: gcn_mae, infogcn_jepa, interaction_dino, etc.
- Unsupervised: bsoid, moseq, subtle, behavemae, clustering
- External: stgcn_pyskl, ctrgcn_pyskl, msg3d_pyskl, etc.
"""
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


# Lazy-loaded external / unsupervised model constructors
_UNSUPERVISED_MODELS = {
    'bsoid': ('behavior_lab.models.unsupervised.bsoid', 'BSOiD'),
    'b-soid': ('behavior_lab.models.unsupervised.bsoid', 'BSOiD'),
    'moseq': ('behavior_lab.models.unsupervised.moseq', 'KeypointMoSeq'),
    'keypoint_moseq': ('behavior_lab.models.unsupervised.moseq', 'KeypointMoSeq'),
    'subtle': ('behavior_lab.models.unsupervised.subtle_wrapper', 'SUBTLE'),
    'behavemae': ('behavior_lab.models.unsupervised.behavemae', 'BehaveMAE'),
    'behave_mae': ('behavior_lab.models.unsupervised.behavemae', 'BehaveMAE'),
}

_EXTERNAL_MODELS = {
    'stgcn_pyskl': ('behavior_lab.models.external.pyskl', 'PySKLModel', 'stgcn'),
    'stgcn++_pyskl': ('behavior_lab.models.external.pyskl', 'PySKLModel', 'stgcn++'),
    'ctrgcn_pyskl': ('behavior_lab.models.external.pyskl', 'PySKLModel', 'ctrgcn'),
    'aagcn_pyskl': ('behavior_lab.models.external.pyskl', 'PySKLModel', 'aagcn'),
    'msg3d_pyskl': ('behavior_lab.models.external.pyskl', 'PySKLModel', 'msg3d'),
    'dgstgcn_pyskl': ('behavior_lab.models.external.pyskl', 'PySKLModel', 'dgstgcn'),
    'poseconv3d_pyskl': ('behavior_lab.models.external.pyskl', 'PySKLModel', 'poseconv3d'),
}


def _lazy_import(module_path: str, class_name: str):
    """Import a class from a module path lazily."""
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_model(name: str, **kwargs):
    """Get any model by name.

    Supervised graph: 'infogcn', 'stgcn', 'agcn'
    Supervised sequence: 'lstm', 'mlp', 'transformer', 'rule_based'
    SSL: 'gcn_dino', 'infogcn_jepa', 'interaction_mae', etc.
    Unsupervised: 'bsoid', 'moseq', 'subtle', 'behavemae'
    External (PySKL): 'stgcn_pyskl', 'ctrgcn_pyskl', 'msg3d_pyskl', etc.
    """
    name_lower = name.lower().replace('-', '_')

    # SSL models
    if name_lower in SSL_REGISTRY:
        return SSL_REGISTRY[name_lower](**kwargs)

    # Graph models (our implementations)
    graph_models = {
        'infogcn': InfoGCN,
        'infogcn_interaction': InfoGCN_Interaction,
        'stgcn': STGCN,
        'agcn': AGCN,
    }
    if name_lower in graph_models:
        kw = _remap_infogcn_kwargs(kwargs) if 'infogcn' in name_lower else kwargs
        return graph_models[name_lower](**kw)

    # Sequence models
    seq_names = ('lstm', 'mlp', 'transformer', 'rule_based', 'rule', 'baseline')
    if name_lower in seq_names:
        return get_action_classifier(name_lower, **kwargs)

    # Unsupervised models (lazy import — heavy deps)
    if name_lower in _UNSUPERVISED_MODELS:
        module_path, class_name = _UNSUPERVISED_MODELS[name_lower]
        cls = _lazy_import(module_path, class_name)
        # BehaveMAE uses from_pretrained, others use __init__
        if name_lower in ('behavemae', 'behave_mae') and 'checkpoint_path' in kwargs:
            return cls.from_pretrained(**kwargs)
        return cls(**kwargs)

    # External models via PySKL (lazy import)
    if name_lower in _EXTERNAL_MODELS:
        module_path, class_name, preset_name = _EXTERNAL_MODELS[name_lower]
        cls = _lazy_import(module_path, class_name)
        if 'checkpoint_path' in kwargs and 'config_path' in kwargs:
            return cls.from_checkpoint(
                kwargs.pop('config_path'), kwargs.pop('checkpoint_path'),
                device=kwargs.pop('device', 'cpu'))
        return cls.from_config(model_name=preset_name, **kwargs)

    # Collect all available model names
    available = (
        list(SSL_REGISTRY.keys()) +
        list(graph_models.keys()) +
        list(seq_names) +
        list(_UNSUPERVISED_MODELS.keys()) +
        list(_EXTERNAL_MODELS.keys())
    )
    raise ValueError(f"Unknown model: {name}. Available: {sorted(set(available))}")


def list_models() -> dict:
    """List all available model names grouped by category."""
    return {
        'graph': ['infogcn', 'infogcn_interaction', 'stgcn', 'agcn'],
        'sequence': ['mlp', 'lstm', 'transformer', 'rule_based'],
        'ssl': sorted(SSL_REGISTRY.keys()),
        'unsupervised': sorted(_UNSUPERVISED_MODELS.keys()),
        'external': sorted(_EXTERNAL_MODELS.keys()),
    }
