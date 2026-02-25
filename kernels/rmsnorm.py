import nkipy.core.typing as nt
import numpy as np
from nkipy.core.tensor_apis import rms_norm


def rmsnorm_disable(x: nt.tensor, weight: nt.tensor, eps: float, is_neuronpy=True):
    original_dtype = x.dtype
    x = x.astype(np.float32)
    weight = weight.astype(np.float32)
    if is_neuronpy:
        # TODO: fix
        return rms_norm(data=x, weight=weight, epsilon=eps).astype(original_dtype)
    else:
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        y = x / rms * weight
        return y.astype(original_dtype)

def rmsnorm(
    x: nt.tensor, 
    weight: nt.tensor, 
    eps: float, 
    is_neuronpy: bool,
):
    # Use float32 to reduce numerical error
    compute_dtype = np.float32
    """
    z: Array["B, L or 1, 1"] = (x**2).mean(-1, keepdims=True) + self.eps
    z: Array["B, L or 1, D"] = x / np.sqrt(z)
    ret = z * self.weight
    """
    original_dtype = x.dtype
    x = x.astype(compute_dtype)
    weight = weight.astype(compute_dtype)
    z = np.square(x)
    if is_neuronpy:
        # FIXME:
        # if this `z` tensor is on PSUM, it might trigger
        # In `codegenPartitionReduceOp` we have `assert inst.op.op != np.mean, 'There is not reduce mean!'`
        # z = np.mean(z, axis=-1, keepdims=True)

        # FIXME: this is another workaround because there is no mean reduction on partition dim
        z = np.divide(z, x.shape[-1]).astype(x.dtype)
        z = np.sum(z, axis=-1, keepdims=True)
    else:
        z = np.mean(z, axis=-1, keepdims=True)
    # with padding
    # z = z * (x.shape[-1] / unpadded_hidden_size)
    z = (z + eps).astype(x.dtype)
    z = x / np.sqrt(z)

    res = z * weight
    res = res.astype(original_dtype)
    return res
