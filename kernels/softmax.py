import nkipy.core.typing as nt
import numpy as np
from nkipy.core.tensor_apis import softmax as softmax_op


def softmax(x: nt.tensor, is_neuronpy: bool):
    original_dtype = x.dtype
    x = x.astype(np.float32)
    if False:
        # FIXME: softmax_op doesn't work, seems to be ignoring sink
        return softmax_op(x).astype(original_dtype)
    else:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(original_dtype)
