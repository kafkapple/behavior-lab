"""Dataset loaders with factory registry."""
from __future__ import annotations

from typing import Any

from .calms21 import CalMS21Loader
from .rat7m import Rat7MLoader
from .ntu_rgbd import NTURGBDLoader
from .nwucla import NWUCLALoader
from .subtle import SUBTLELoader
from .shank3ko import Shank3KOLoader
from .mabe22 import MABe22Loader

LOADER_REGISTRY: dict[str, type] = {
    "calms21": CalMS21Loader,
    "calms21_mouse": CalMS21Loader,
    "rat7m": Rat7MLoader,
    "ntu": NTURGBDLoader,
    "ntu_rgbd": NTURGBDLoader,
    "ntu60": NTURGBDLoader,
    "ntu120": NTURGBDLoader,
    "nwucla": NWUCLALoader,
    "nw_ucla": NWUCLALoader,
    "subtle": SUBTLELoader,
    "subtle_mouse": SUBTLELoader,
    "shank3ko": Shank3KOLoader,
    "shank3ko_mouse": Shank3KOLoader,
    "mabe22": MABe22Loader,
    "mabe": MABe22Loader,
}


def get_loader(name: str, **kwargs: Any):
    """Factory function to get a dataset loader by name.

    Args:
        name: Dataset identifier (e.g., 'calms21', 'rat7m', 'ntu', 'nwucla')
        **kwargs: Passed to the loader constructor

    Returns:
        Instantiated loader
    """
    key = name.lower()
    if key not in LOADER_REGISTRY:
        available = sorted(set(LOADER_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")
    return LOADER_REGISTRY[key](**kwargs)


def register_loader(name: str, loader_cls: type) -> None:
    """Register a custom dataset loader."""
    LOADER_REGISTRY[name.lower()] = loader_cls


__all__ = [
    "CalMS21Loader",
    "Rat7MLoader",
    "NTURGBDLoader",
    "NWUCLALoader",
    "SUBTLELoader",
    "Shank3KOLoader",
    "MABe22Loader",
    "get_loader",
    "register_loader",
    "LOADER_REGISTRY",
]
