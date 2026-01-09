# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from nkipy.core.specs import CommonTypes, KernelSpec, ShapeSpec, TensorInputSpec


def nkipy_kernel_func(arg0_1, arg1_1):
    # arg0_1: f32[32]
    # arg1_1: i64[1, 7]

    # unsqueeze: f32[1, 32] = torch.ops.aten.unsqueeze.default(arg0_1, 0)
    unsqueeze = np.expand_dims(arg0_1, 0)
    # slice_1: f32[1, 32] = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 9223372036854775807)
    slice_1 = unsqueeze[:, 0:]
    # unsqueeze_1: f32[1, 32, 1] = torch.ops.aten.unsqueeze.default(slice_1, 2)
    unsqueeze_1 = np.expand_dims(slice_1, 2)
    # expand: f32[1, 32, 1] = torch.ops.aten.expand.default(unsqueeze_1, [1, -1, 1])
    expand = np.broadcast_to(unsqueeze_1, [1, 32, 1])

    # slice_2: i64[1, 7] = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807)
    slice_2 = arg1_1[0:]
    # unsqueeze_2: i64[1, 1, 7] = torch.ops.aten.unsqueeze.default(slice_2, 1)
    unsqueeze_2 = np.expand_dims(slice_2, 1)
    # slice_3: i64[1, 1, 7] = torch.ops.aten.slice.Tensor(unsqueeze_2, 2, 0, 9223372036854775807)
    slice_3 = unsqueeze_2[:, :, 0:]
    # _to_copy: f32[1, 1, 7] = torch.ops.aten._to_copy.default(slice_3, dtype = torch.float32)
    _to_copy = slice_3.astype(np.float32)

    # expand_1: f32[1, 32, 1] = torch.ops.aten.expand.default(expand, [1, 32, 1])
    expand_1 = np.broadcast_to(expand, [1, 32, 1])
    # view: f32[1, 32, 1] = torch.ops.aten.view.default(expand_1, [1, 32, 1])
    view = np.reshape(expand_1, [1, 32, 1])
    # expand_2: f32[1, 1, 7] = torch.ops.aten.expand.default(_to_copy, [1, 1, 7])
    expand_2 = np.broadcast_to(_to_copy, [1, 1, 7])
    # view_1: f32[1, 1, 7] = torch.ops.aten.view.default(expand_2, [1, 1, 7])
    view_1 = np.reshape(expand_2, [1, 1, 7])
    # bmm: f32[1, 32, 7] = torch.ops.aten.bmm.default(view, view_1)
    bmm = np.matmul(view, view_1)
    # view_2: f32[1, 32, 7] = torch.ops.aten.view.default(bmm, [1, 32, 7])
    view_2 = np.reshape(bmm, [1, 32, 7])
    # permute: f32[1, 7, 32] = torch.ops.aten.permute.default(view_2, [0, 2, 1])
    permute = np.transpose(view_2, [0, 2, 1])

    # cat: f32[1, 7, 64] = torch.ops.aten.cat.default([permute, permute], -1)
    cat = np.concatenate([permute, permute], -1)

    # cos: f32[1, 7, 64] = torch.ops.aten.cos.default(cat)
    cos = np.cos(
        cat,
    )

    # sin: f32[1, 7, 64] = torch.ops.aten.sin.default(cat)
    sin = np.sin(
        cat,
    )

    # mul: f32[1, 7, 64] = torch.ops.aten.mul.Tensor(cos, 1.0)
    mul = np.multiply(cos, 1.0)

    # mul_1: f32[1, 7, 64] = torch.ops.aten.mul.Tensor(sin, 1.0)
    mul_1 = np.multiply(sin, 1.0)
    return (mul, mul_1)


arg0_1 = np.random.uniform(0, 1, (32,)).astype(np.float32)
arg1_1 = np.random.randint(0, 16, (1, 7)).astype(np.int32)

# Test values
arg_0_1 = np.random.uniform(0, 1, (32,)).astype(np.float32)
arg_1_1 = np.random.randint(0, 16, (1, 7)).astype(np.int32)

ROTARY_EMBEDDING_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[32], default=(32,)),
    dtype_spec=CommonTypes.FLOATS,
    description="Rotary embedding base tensor",
    default_value=arg_0_1,
)

POSITION_IDS_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 7], default=(1, 7)),
    dtype_spec=CommonTypes.INTS,
    description="Position indices",
    default_value=arg_1_1,
)

# Define the kernel spec
kernel_specs = [
    KernelSpec(
        function=nkipy_kernel_func,
        inputs=[
            ROTARY_EMBEDDING_SPEC,
            POSITION_IDS_SPEC,
        ],
        description="Rotary Position Embedding generation kernel",
        is_pure_numpy=True,
    )
]
