from ._spike import (
    ModelTensorInfo,
    NonBlockExecResult,
    NonBlockTensorReadResult,
    NonBlockTensorWriteResult,
    NrtError,
    NrtModel,
    NrtTensor,
    NrtTensorSet,
    Spike,
    SpikeError,
    SystemTraceSession,
    TensorMetadata,
)
from .profiler_adapter import SpikeProfiler
from .spike_async import SpikeAsync, SpikeStream
from .spike_model import BenchmarkResult, SpikeModel
from .spike_singleton import configure, get_spike_singleton, reset
from .spike_tensor import SpikeTensor

# Type alias for convenience
NonBlockResult = NonBlockTensorReadResult | NonBlockTensorWriteResult | NonBlockExecResult

__all__ = [
    "SpikeModel",
    "SpikeTensor",
    "SpikeProfiler",
    "SpikeAsync",
    "SpikeStream",
    "Spike",
    "configure",
    "reset",
    "get_spike_singleton",
    "NrtError",
    "SpikeError",
    "SystemTraceSession",
    "BenchmarkResult",
    "ModelTensorInfo",
    "NrtModel",
    "NrtTensor",
    "NrtTensorSet",
    "TensorMetadata",
    "NonBlockTensorReadResult",
    "NonBlockTensorWriteResult",
    "NonBlockExecResult",
    "NonBlockResult",
]
