import importlib
import pathlib
import sys

_build_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "build"
if _build_dir.is_dir() and str(_build_dir) not in sys.path:
    sys.path.insert(0, str(_build_dir))

try:
    _relay = importlib.import_module("_relay")
except ImportError:
    _relay = None

from .endpoint import RankEndpoint
from .transfer import (
    WeightServer,
    collect_weight_buffers,
    fetch_tok_embedding,
    preconnect_to_peer,
    preregister_weights,
    push_to_peer,
    push_weights_to_peer,
    rank_endpoint,
    receive_from_peer,
    receive_weights,
)

__all__ = [
    "RankEndpoint",
    "WeightServer",
    "collect_weight_buffers",
    "fetch_tok_embedding",
    "preconnect_to_peer",
    "preregister_weights",
    "push_to_peer",
    "push_weights_to_peer",
    "rank_endpoint",
    "receive_from_peer",
    "receive_weights",
]

__version__ = "0.0.1.post4"
