# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Array creation operations: zeros, full, zeros_like, empty_like, full_like, ones_like"""

import numpy as np

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# Primitive creation ops
# -----------------------------------------------------------------------------
zeros = Op("zeros")
full = Op("full")
constant = Op("constant")
zeros_like = Op("zeros_like")
ones_like = Op("ones_like")
empty_like = Op("empty_like")
full_like = Op("full_like")
tril = Op("tril")
triu = Op("triu")
diag = Op("diag")


# -----------------------------------------------------------------------------
# CPU implementations (needed for non-tracing execution)
# -----------------------------------------------------------------------------


@zeros.impl("cpu")
def _zeros_cpu(shape, dtype):
    return np.zeros(shape, dtype=dtype)


@zeros_like.impl("cpu")
def _zeros_like_cpu(x, dtype=None):
    return np.zeros_like(x, dtype=dtype)


@ones_like.impl("cpu")
def _ones_like_cpu(x, dtype=None):
    return np.ones_like(x, dtype=dtype)


@empty_like.impl("cpu")
def _empty_like_cpu(x, dtype=None):
    return np.empty_like(x, dtype=dtype)


@full_like.impl("cpu")
def _full_like_cpu(x, fill_value, dtype=None):
    return np.full_like(x, fill_value, dtype=dtype)


@full.impl("cpu")
def _full_cpu(shape, fill_value, dtype):
    return np.full(shape, fill_value, dtype=dtype)


@constant.impl("cpu")
def _constant_cpu(value, dtype=None):
    return np.asarray(value, dtype=dtype)
