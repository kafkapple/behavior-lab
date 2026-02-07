"""Graph-based models: self-implemented GCN family + external PySKL wrappers."""
from .infogcn import InfoGCN, InfoGCN_Interaction
from .baselines import STGCN, AGCN
from .pyskl import PySKLModel
