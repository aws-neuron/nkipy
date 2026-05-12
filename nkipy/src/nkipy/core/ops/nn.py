# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neural network operations: softmax, topk, rms_norm"""

import numpy as np

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# softmax
# -----------------------------------------------------------------------------
softmax = Op("softmax")

# -----------------------------------------------------------------------------
# topk
# -----------------------------------------------------------------------------
topk = Op("topk")

# -----------------------------------------------------------------------------
# rms_norm
# -----------------------------------------------------------------------------
rms_norm = Op("rms_norm")


@topk.impl("cpu")
def _topk_cpu(x, k, axis=0, is_ascend=False, out=None, dtype=None):
    """Top-k operation (CPU)."""
    if axis < 0:
        axis = x.ndim + axis

    if is_ascend:
        indices = np.argpartition(x, k - 1, axis=axis)
        indices = np.take(indices, range(k), axis=axis)
        values = np.take_along_axis(x, indices, axis=axis)
        sort_indices = np.argsort(values, axis=axis)
        indices = np.take_along_axis(indices, sort_indices, axis=axis)
        values = np.take_along_axis(values, sort_indices, axis=axis)
    else:
        indices = np.argpartition(-x, k - 1, axis=axis)
        indices = np.take(indices, range(k), axis=axis)
        values = np.take_along_axis(x, indices, axis=axis)
        sort_indices = np.argsort(-values, axis=axis)
        indices = np.take_along_axis(indices, sort_indices, axis=axis)
        values = np.take_along_axis(values, sort_indices, axis=axis)

    return values, indices.astype(np.uint32)
