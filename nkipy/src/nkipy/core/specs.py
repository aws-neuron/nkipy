# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Specification of a kernel including shapes and types"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np


@dataclass
class ShapeSpec:
    """Specification for tensor shapes.

    Each dimension can be:
    - int: exact size required
    - tuple(int|None, int|None): size range, None means unlimited
    - None: any size allowed

    Examples:
        ShapeSpec(dims=[None, 32])  # (*, 32)
        ShapeSpec(dims=[None, (16, 64)])  # (*, 16-64)
        ShapeSpec(dims=[1, None, (16, None)])  # (1, *, 16-*)
        ShapeSpec(dims=[None, (None, 64)])  # (*, *-64)
    """

    dims: List[Union[int, Tuple[Optional[int], Optional[int]], None]]
    default: Tuple[int, ...]

    def is_valid(self, shape: Tuple[int, ...]) -> bool:
        if len(shape) != len(self.dims):
            return False

        for actual, expected in zip(shape, self.dims):
            if expected is None:
                continue
            if isinstance(expected, int) and actual != expected:
                return False
            if isinstance(expected, tuple):
                min_size, max_size = expected
                if min_size is not None and actual < min_size:
                    return False
                if max_size is not None and actual > max_size:
                    return False
        return True


@dataclass
class TypeSpec:
    """Specification for types"""

    allowed: List[Type]
    default: Type

    def is_valid(self, dtype: Type) -> bool:
        return dtype in self.allowed


@dataclass(kw_only=True)
class InputSpec:
    dtype_spec: TypeSpec
    description: str = ""
    default_value: Any = None


@dataclass(kw_only=True)
class ScalarInputSpec(InputSpec):
    description: str = "Scalar input"


@dataclass(kw_only=True)
class TensorInputSpec(InputSpec):
    shape_spec: ShapeSpec
    description: str = "Tensor input"


@dataclass
class KernelSpec:
    """Full specification of a kernel"""

    function: Any
    inputs: List[InputSpec]
    is_pure_numpy: bool = True
    description: str = ""


# Common type specifications
class CommonTypes:
    # FIXME: add bf16 here
    FLOATS = TypeSpec(allowed=[np.float32, np.float16], default=np.float32)
    INTS = TypeSpec(allowed=[np.int8, np.int16, np.int32, np.int64], default=np.int32)
    BOOL = TypeSpec(allowed=[bool], default=bool)
