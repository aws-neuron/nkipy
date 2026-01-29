import neuronxcc.nki.language as nl
import torch

from torch_to_nkipy import NKIOpRegistry, init_nkipy_backend

torch.manual_seed(0)

init_nkipy_backend()


@NKIOpRegistry.register("mylib::add_custom_op")
def nki_tensor_add_kernel(a_input, b_input):
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    ix = nl.arange(128)[:, None]
    iy = nl.arange(512)[None, :]
    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])
    c_tile = a_tile + b_tile
    nl.store(c_output[ix, iy], value=c_tile)
    return c_output


@torch.library.custom_op("mylib::add_custom_op", mutates_args=())
def nki_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nki_tensor_add_kernel(a, b)


@nki_add.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)


@torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
def nki_func(a, b):
    return nki_add(a, b)


def cpu_func(a, b):
    return a + b


a = torch.randn(128, 512)
b = torch.randn(128, 512)
out_nki = nki_func(a.to("nkipy"), b.to("nkipy"))

out_cpu = cpu_func(a, b)
print(f"Is nki result close to cpu result? {torch.allclose(out_nki.cpu(), out_cpu)}")
