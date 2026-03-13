"""NKIPy Spike Runtime C++ bindings"""

from collections.abc import Mapping, Sequence
from typing import overload

from numpy.typing import NDArray


class SpikeRuntimeError(RuntimeError):
    pass

class SpikeError(SpikeRuntimeError):
    pass

class NrtError(SpikeRuntimeError):
    pass

class SystemTraceSession:
    def __init__(self, core_id: int | None = None) -> None:
        """
        Create a trace session. Trace the given core_id or if core_id is omitted, traces all visible NeuronCores.
        """

    def stop(self) -> None:
        """Stop tracing. Called automatically on __exit__."""

    def fetch_events_json(self) -> str:
        """
        Fetch events as JSON string, consumes events from the system trace ring buffer
        """

    def drain_events(self) -> None:
        """Discard events in the buffer"""

    def __enter__(self) -> SystemTraceSession: ...

    def __exit__(self, *args) -> None: ...

class TensorMetadata:
    @property
    def size(self) -> int:
        """Tensor size in bytes"""

    @property
    def dtype(self) -> str:
        """Data type as string"""

    @property
    def shape(self) -> list[int]:
        """Tensor shape as list"""

    def __repr__(self) -> str: ...

class ModelTensorInfo:
    @property
    def inputs(self) -> dict[str, TensorMetadata]:
        """Input tensor metadata"""

    @property
    def outputs(self) -> dict[str, TensorMetadata]:
        """Output tensor metadata"""

    def __repr__(self) -> str: ...

class NrtTensor:
    @property
    def core_id(self) -> int:
        """Logical NeuronCore ID"""

    @property
    def size(self) -> int:
        """Tensor size in bytes"""

    @property
    def name(self) -> str:
        """Tensor name"""

    @property
    def va(self) -> int:
        """CPU-accessible virtual address of device HBM memory"""

class NrtModel:
    @property
    def neff_path(self) -> str:
        """NEFF file path"""

    @property
    def core_id(self) -> int:
        """Core ID"""

    @property
    def rank_id(self) -> int:
        """Rank ID"""

    @property
    def world_size(self) -> int:
        """World size"""

    @property
    def is_collective(self) -> bool:
        """Is collective model"""

    def __repr__(self) -> str: ...

class NrtTensorSet:
    pass

class NonBlockTensorReadResult:
    @property
    def id(self) -> int: ...

    @property
    def data(self) -> bytes | NDArray: ...

    @property
    def err(self) -> "spike::SpikeError" | "spike::NrtError" | None: ...

class NonBlockTensorWriteResult:
    @property
    def id(self) -> int: ...

    @property
    def err(self) -> "spike::SpikeError" | "spike::NrtError" | None: ...

class NonBlockExecResult:
    @property
    def id(self) -> int: ...

    @property
    def err(self) -> "spike::SpikeError" | "spike::NrtError" | None: ...

class Spike:
    def __init__(self, verbose_level: int = 0) -> None:
        """Initialize Spike with verbose level"""

    @staticmethod
    def get_visible_neuron_core_count() -> int:
        """Get the number of visible NeuronCores"""

    def close(self) -> int:
        """Close the NRT runtime"""

    def load_model(self, neff_file: str, core_id: int = 0, cc_enabled: bool = False, rank_id: int = 0, world_size: int = 1) -> NrtModel:
        """Load a model from NEFF file"""

    def unload_model(self, model: NrtModel) -> None:
        """Unload a model"""

    def execute(self, model: NrtModel, inputs: Mapping[str, NrtTensor], outputs: Mapping[str, NrtTensor], ntff_name: str | None = None, save_trace: bool = False) -> None:
        """Execute a model with given inputs and outputs"""

    def allocate_tensor(self, size: int, core_id: int = 0, name: str | None = None) -> NrtTensor:
        """Allocate a tensor on device"""

    def slice_from_tensor(self, source: NrtTensor, offset: int = 0, size: int = 0, name: str | None = None) -> NrtTensor:
        """Create a tensor slice from another tensor"""

    def free_tensor(self, tensor: NrtTensor) -> None:
        """Free a tensor"""

    def tensor_write(self, tensor: NrtTensor, data: bytes, offset: int = 0) -> None:
        """Write bytes data to tensor"""

    def tensor_read(self, tensor: NrtTensor, offset: int = 0, size: int = 0) -> bytes:
        """Read data from tensor as bytes"""

    def tensor_write_from_pybuffer(self, tensor: NrtTensor, buffer: object, offset: int = 0) -> None:
        """
        Write data from Python buffer protocol object (bytes, bytearray, memoryview, etc.) to tensor
        """

    def tensor_read_to_pybuffer(self, tensor: NrtTensor, buffer: object, offset: int = 0, size: int = 0) -> None:
        """
        Read data from tensor to Python buffer protocol object (bytearray, memoryview, etc.)
        """

    def init_nonblock(self) -> None:
        """Initialize for nonblocking operations"""

    @overload
    def tensor_read_nonblock(self, tensor: NrtTensor, offset: int = 0, size: int = 0) -> int:
        """Read data from tensor as bytes nonblockingly"""

    @overload
    def tensor_read_nonblock(self, tensor: NrtTensor, dest: NDArray, offset: int = 0, size: int = 0) -> int:
        """Read data from tensor into the provided destination nonblockingly"""

    @overload
    def tensor_write_nonblock(self, tensor: NrtTensor, data: bytes, offset: int = 0) -> int:
        """Write bytes data to tensor nonblockingly"""

    @overload
    def tensor_write_nonblock(self, tensor: NrtTensor, data: NDArray, offset: int = 0) -> int:
        """Write ndarray data to tensor nonblockingly"""

    @overload
    def tensor_write_nonblock(self, tensor: NrtTensor, data: int, size: int, offset: int = 0) -> int:
        """Write raw pointer data to tensor nonblockingly"""

    def tensor_write_nonblock_batched_prepare(self, tensors: Sequence[NrtTensor], data_objs: Sequence[NDArray], offsets: Sequence[int] | None = None) -> int:
        """Prepare a batched tensor write"""

    def tensor_write_nonblock_batched_start(self, batch_id: int) -> int:
        """Start a prepared batched tensor write"""

    def tensor_read_nonblock_batched_prepare(self, tensors: Sequence[NrtTensor], dests: Sequence[NDArray], offsets: Sequence[int] | None = None, sizes: Sequence[int] | None = None) -> int:
        """Prepare a batched tensor read"""

    def tensor_read_nonblock_batched_start(self, batch_id: int) -> int:
        """Start a prepared batched tensor read"""

    def execute_nonblock(self, model: NrtModel, inputs: NrtTensorSet, outputs: NrtTensorSet, ntff_name: str | None = None, save_trace: bool = False) -> int:
        """Execute a model with given inputs and outputs nonblockingly"""

    def try_poll(self) -> NonBlockTensorReadResult | NonBlockTensorWriteResult | NonBlockExecResult | None:
        """Try to poll for nonblocking results"""

    def create_tensor_set(self, tensors: Mapping[str, NrtTensor]) -> NrtTensorSet:
        """Create a tensor set with the tensors"""

    def wrap_model(self, ptr: int) -> NrtModel:
        """Wrap an existing NRT model pointer"""

    def wrap_tensor(self, ptr: int) -> NrtTensor:
        """Wrap an existing NRT tensor pointer"""

    def wrap_tensor_set(self, ptr: int) -> NrtTensorSet:
        """Wrap an existing NRT tensor set pointer"""

    def get_tensor_info(self, model: NrtModel) -> ModelTensorInfo:
        """Get tensor information for a model"""
