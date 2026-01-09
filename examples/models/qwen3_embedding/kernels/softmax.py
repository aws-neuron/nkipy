import numpy as np


def softmax(x):
    original_dtype = x.dtype
    x = x.astype(np.float32)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(original_dtype)
