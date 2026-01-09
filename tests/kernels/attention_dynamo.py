# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Attention Layer (prefill) generated from torch dynamo graph

import numpy as np
from nkipy.core.specs import CommonTypes, KernelSpec, ShapeSpec, TensorInputSpec


def nkipy_kernel_func(
    arg0_1,
    arg1_1,
    arg2_1,
    arg3_1,
    arg4_1,
    arg5_1,
    arg6_1,
    arg7_1,
    arg8_1,
    arg9_1,
    arg10_1,
):
    permute = np.transpose(arg1_1, [1, 0])
    view = np.reshape(arg0_1, [7, 2048])
    mm = np.matmul(view, permute)
    view_1 = np.reshape(mm, [1, 7, 2048])
    view_2 = np.reshape(view_1, [1, 7, -1, 64])
    permute_1 = np.transpose(view_2, [0, 2, 1, 3])
    permute_2 = np.transpose(arg2_1, [1, 0])
    view_3 = np.reshape(arg0_1, [7, 2048])
    mm_1 = np.matmul(view_3, permute_2)
    view_4 = np.reshape(mm_1, [1, 7, 512])
    view_5 = np.reshape(view_4, [1, 7, -1, 64])
    permute_3 = np.transpose(view_5, [0, 2, 1, 3])
    permute_4 = np.transpose(arg3_1, [1, 0])
    view_6 = np.reshape(arg0_1, [7, 2048])
    mm_2 = np.matmul(view_6, permute_4)
    view_7 = np.reshape(mm_2, [1, 7, 512])
    view_8 = np.reshape(view_7, [1, 7, -1, 64])
    permute_5 = np.transpose(view_8, [0, 2, 1, 3])
    unsqueeze = np.expand_dims(arg4_1, 1)
    unsqueeze_1 = np.expand_dims(arg5_1, 1)
    mul = np.multiply(permute_1, unsqueeze)
    slice_1 = permute_1[:, :, :, 0:32]
    slice_2 = permute_1[:, :, :, 32:]
    neg = np.negative(slice_2)
    cat = np.concatenate([neg, slice_1], -1)
    mul_1 = np.multiply(cat, unsqueeze_1)
    add = np.add(mul, mul_1)
    mul_2 = np.multiply(permute_3, unsqueeze)
    slice_3 = permute_3[:, :, :, 0:32]
    slice_4 = permute_3[:, :, :, 32:]
    neg_1 = np.negative(slice_4)
    cat_1 = np.concatenate([neg_1, slice_3], -1)
    mul_3 = np.multiply(cat_1, unsqueeze_1)
    add_1 = np.add(mul_2, mul_3)
    index_put = np.copy(arg7_1)

    # arg6_1 = np.expand_dims(arg6_1, axis=(0, 1, 3))
    # np.put_along_axis(index_put, arg6_1, add_1, 2)
    index_put[:, :, arg6_1] = add_1
    index_put_1 = np.copy(arg8_1)
    # np.put_along_axis(index_put_1, arg6_1, permute_5, 2)
    index_put_1[:, :, arg6_1] = permute_5
    # slice_9 = index_put[0:]
    # slice_10 = slice_9[:, 0:]
    slice_10 = index_put[0:, 0:]
    unsqueeze_3 = np.expand_dims(slice_10, 2)
    # slice_11 = unsqueeze_3[:, :, :, 0:]
    # slice_12 = slice_11[:, :, :, :, 0:]
    slice_12 = unsqueeze_3[:, :, :, :, 0:]
    expand_1 = np.broadcast_to(slice_12, [1, 8, 4, 16, 64])
    clone = np.copy(
        expand_1,
    )
    view_9 = np.reshape(clone, [1, 32, 16, 64])
    # slice_17 = index_put_1[0:]
    # slice_18 = slice_17[:, 0:]
    slice_18 = index_put_1[0:, 0:]
    unsqueeze_5 = np.expand_dims(slice_18, 2)
    # slice_19 = unsqueeze_5[:, :, :, 0:]
    # slice_20 = slice_19[:, :, :, :, 0:]
    slice_20 = unsqueeze_5[:, :, :, :, 0:]
    expand_3 = np.broadcast_to(slice_20, [1, 8, 4, 16, 64])
    clone_1 = np.copy(
        expand_3,
    )
    view_10 = np.reshape(clone_1, [1, 32, 16, 64])
    permute_6 = np.transpose(view_9, [0, 1, 3, 2])
    expand_4 = np.broadcast_to(add, [1, 32, 7, 64])
    view_11 = np.reshape(expand_4, [32, 7, 64])
    expand_5 = np.broadcast_to(permute_6, [1, 32, 64, 16])
    view_12 = np.reshape(expand_5, [32, 64, 16])
    bmm = np.matmul(view_11, view_12)
    view_13 = np.reshape(bmm, [1, 32, 7, 16])
    mul_4 = np.multiply(view_13, 0.125)
    # slice_21 = arg9_1[0:]
    # slice_22 = slice_21[:, 0:]
    # slice_23 = slice_22[:, :, 0:]
    slice_23 = arg9_1[0:, 0:, 0:]
    add_2 = np.add(mul_4, slice_23)
    add_2_np_0 = np.max(add_2, axis=-1, keepdims=True)

    # Expand subtract and divide
    add_2_np_1 = np.exp(np.subtract(add_2, add_2_np_0))
    _softmax = np.divide(add_2_np_1, np.sum(add_2_np_1, axis=-1, keepdims=True))
    clone_2 = np.copy(
        _softmax,
    )
    expand_6 = np.broadcast_to(clone_2, [1, 32, 7, 16])
    view_14 = np.reshape(expand_6, [32, 7, 16])
    expand_7 = np.broadcast_to(view_10, [1, 32, 16, 64])
    view_15 = np.reshape(expand_7, [32, 16, 64])
    bmm_1 = np.matmul(view_14, view_15)
    view_16 = np.reshape(bmm_1, [1, 32, 7, 64])
    permute_7 = np.transpose(view_16, [0, 2, 1, 3])
    clone_3 = np.copy(
        permute_7,
    )
    view_17 = np.reshape(clone_3, [1, 7, -1])
    permute_8 = np.transpose(arg10_1, [1, 0])
    view_18 = np.reshape(view_17, [7, 2048])
    mm_3 = np.matmul(view_18, permute_8)
    view_19 = np.reshape(mm_3, [1, 7, 2048])
    return view_19, clone_2


# TODO: verify whether the spec matches the actual functionality
arg_0_1 = np.random.uniform(0, 1, (1, 7, 2048)).astype(np.float32)
arg_1_1 = np.random.uniform(0, 1, (2048, 2048)).astype(np.float32)
arg_2_1 = np.random.uniform(0, 1, (512, 2048)).astype(np.float32)
arg_3_1 = np.random.uniform(0, 1, (512, 2048)).astype(np.float32)
arg_4_1 = np.random.uniform(0, 1, (1, 7, 64)).astype(np.float32)
arg_5_1 = np.random.uniform(0, 1, (1, 7, 64)).astype(np.float32)
arg_6_1 = np.random.randint(0, 16, (7,)).astype(np.int32)
arg_7_1 = np.random.uniform(0, 1, (1, 8, 16, 64)).astype(np.float32)
arg_8_1 = np.random.uniform(0, 1, (1, 8, 16, 64)).astype(np.float32)
arg_9_1 = np.random.uniform(0, 1, (1, 1, 7, 16)).astype(np.float32)
arg_10_1 = np.random.uniform(0, 1, (2048, 2048)).astype(np.float32)

INPUT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 7, 2048], default=(1, 7, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Input tensor",
    default_value=arg_0_1,
)

QKV_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[2048, 2048], default=(2048, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Query weight matrix",
    default_value=arg_1_1,
)

KEY_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[512, 2048], default=(512, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Key weight matrix",
    default_value=arg_2_1,
)

VALUE_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[512, 2048], default=(512, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Value weight matrix",
    default_value=arg_3_1,
)

ROTARY_POS1_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 7, 64], default=(1, 7, 64)),
    dtype_spec=CommonTypes.FLOATS,
    description="First rotary positional embedding",
    default_value=arg_4_1,
)

ROTARY_POS2_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 7, 64], default=(1, 7, 64)),
    dtype_spec=CommonTypes.FLOATS,
    description="Second rotary positional embedding",
    default_value=arg_5_1,
)

POSITION_IDS_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[7], default=(7,)),
    dtype_spec=CommonTypes.INTS,
    description="Position indices",
    default_value=arg_6_1,
)

KV_CACHE1_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 8, 16, 64], default=(1, 8, 16, 64)),
    dtype_spec=CommonTypes.FLOATS,
    description="First KV cache tensor",
    default_value=arg_7_1,
)

KV_CACHE2_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 8, 16, 64], default=(1, 8, 16, 64)),
    dtype_spec=CommonTypes.FLOATS,
    description="Second KV cache tensor",
    default_value=arg_8_1,
)

ATTN_MASK_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 1, 7, 16], default=(1, 1, 7, 16)),
    dtype_spec=CommonTypes.FLOATS,
    description="Attention mask",
    default_value=arg_9_1,
)

OUTPUT_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[2048, 2048], default=(2048, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Output projection weight matrix",
    default_value=arg_10_1,
)

# Define the kernel spec
kernel_specs = [
    KernelSpec(
        function=nkipy_kernel_func,
        inputs=[
            INPUT_SPEC,
            QKV_WEIGHT_SPEC,
            KEY_WEIGHT_SPEC,
            VALUE_WEIGHT_SPEC,
            ROTARY_POS1_SPEC,
            ROTARY_POS2_SPEC,
            POSITION_IDS_SPEC,
            KV_CACHE1_SPEC,
            KV_CACHE2_SPEC,
            ATTN_MASK_SPEC,
            OUTPUT_WEIGHT_SPEC,
        ],
        description="Attention layer with rotary position embedding and KV cache",
        is_pure_numpy=True,
    )
]
