# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Llama transformer Layer with Attention and MLP blocks generated from torch dynamo graph

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
    arg11_1,
    arg12_1,
    arg13_1,
    arg14_1,
    arg15_1,
    arg16_1,
    arg17_1,
):
    # arg0_1: i64[1, 7]
    # arg1_1: f32[128256, 2048]
    # arg2_1: i64[7]
    # arg3_1: i64[1, 7]
    # arg4_1: f32[1, 8, 16, 64]
    # arg5_1: f32[1, 1, 7, 16]
    # arg6_1: f32[32]
    # arg7_1: f32[2048]
    # arg8_1: f32[2048, 2048]
    # arg9_1: f32[512, 2048]
    # arg10_1: f32[512, 2048]
    # arg11_1: f32[1, 8, 16, 64]
    # arg12_1: f32[2048, 2048]
    # arg13_1: f32[2048]
    # arg14_1: f32[8192, 2048]
    # arg15_1: f32[8192, 2048]
    # arg16_1: f32[2048, 8192]
    # arg17_1: f32[2048]

    # embedding: f32[1, 7, 2048] = torch.ops.aten.embedding.default(arg1_1, arg0_1)
    embedding = np.take(arg1_1, arg0_1, axis=0)

    unsqueeze = np.expand_dims(arg6_1, 0)
    # slice_1: f32[1, 32] = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 9223372036854775807)
    slice_1 = unsqueeze[:, 0:]
    # unsqueeze_1: f32[1, 32, 1] = torch.ops.aten.unsqueeze.default(slice_1, 2)
    unsqueeze_1 = np.expand_dims(slice_1, 2)
    # expand: f32[1, 32, 1] = torch.ops.aten.expand.default(unsqueeze_1, [1, -1, 1])
    expand = np.broadcast_to(unsqueeze_1, [1, 32, 1])

    slice_2 = arg3_1[0:]
    # unsqueeze_2: i64[1, 1, 7] = torch.ops.aten.unsqueeze.default(slice_2, 1)
    unsqueeze_2 = np.expand_dims(slice_2, 1)
    # slice_3: i64[1, 1, 7] = torch.ops.aten.slice.Tensor(unsqueeze_2, 2, 0, 9223372036854775807)
    slice_3 = unsqueeze_2[:, :, 0:]
    # _to_copy: f32[1, 1, 7] = torch.ops.aten._to_copy.default(slice_3, dtype = torch.float32)
    _to_copy = slice_3.astype(np.float32)

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

    cat = np.concatenate([permute, permute], -1)

    cos = np.cos(
        cat,
    )

    sin = np.sin(
        cat,
    )

    mul = np.multiply(cos, 1.0)

    mul_1 = np.multiply(sin, 1.0)

    pow_1 = np.power(embedding, 2)
    # mean: f32[1, 7, 1] = torch.ops.aten.mean.dim(pow_1, [-1], True)
    pow_1_np_0 = np.sum(pow_1, axis=(-1,), keepdims=True)
    mean = np.divide(pow_1_np_0, pow_1.shape[-1])

    add = np.add(mean, 1e-05)
    # rsqrt: f32[1, 7, 1] = torch.ops.aten.rsqrt.default(add)
    add_np_0 = np.sqrt(add)
    rsqrt = np.divide(1, add_np_0)
    # mul_2: f32[1, 7, 2048] = torch.ops.aten.mul.Tensor(embedding, rsqrt)
    mul_2 = np.multiply(embedding, rsqrt)

    mul_3 = np.multiply(arg7_1, mul_2)

    permute_1 = np.transpose(arg8_1, [1, 0])
    # view_3: f32[7, 2048] = torch.ops.aten.view.default(mul_3, [7, 2048])
    view_3 = np.reshape(mul_3, [7, 2048])
    # mm: f32[7, 2048] = torch.ops.aten.mm.default(view_3, permute_1)
    mm = np.matmul(view_3, permute_1)
    # view_4: f32[1, 7, 2048] = torch.ops.aten.view.default(mm, [1, 7, 2048])
    view_4 = np.reshape(mm, [1, 7, 2048])
    # view_5: f32[1, 7, 32, 64] = torch.ops.aten.view.default(view_4, [1, 7, -1, 64])
    view_5 = np.reshape(view_4, [1, 7, -1, 64])
    # permute_2: f32[1, 32, 7, 64] = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3])
    permute_2 = np.transpose(view_5, [0, 2, 1, 3])

    permute_3 = np.transpose(arg9_1, [1, 0])
    # view_6: f32[7, 2048] = torch.ops.aten.view.default(mul_3, [7, 2048])
    view_6 = np.reshape(mul_3, [7, 2048])
    # mm_1: f32[7, 512] = torch.ops.aten.mm.default(view_6, permute_3)
    mm_1 = np.matmul(view_6, permute_3)
    # view_7: f32[1, 7, 512] = torch.ops.aten.view.default(mm_1, [1, 7, 512])
    view_7 = np.reshape(mm_1, [1, 7, 512])
    # view_8: f32[1, 7, 8, 64] = torch.ops.aten.view.default(view_7, [1, 7, -1, 64])
    view_8 = np.reshape(view_7, [1, 7, -1, 64])
    # permute_4: f32[1, 8, 7, 64] = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3])
    permute_4 = np.transpose(view_8, [0, 2, 1, 3])

    permute_5 = np.transpose(arg10_1, [1, 0])
    # view_9: f32[7, 2048] = torch.ops.aten.view.default(mul_3, [7, 2048])
    view_9 = np.reshape(mul_3, [7, 2048])
    # mm_2: f32[7, 512] = torch.ops.aten.mm.default(view_9, permute_5)
    mm_2 = np.matmul(view_9, permute_5)
    # view_10: f32[1, 7, 512] = torch.ops.aten.view.default(mm_2, [1, 7, 512])
    view_10 = np.reshape(mm_2, [1, 7, 512])
    # view_11: f32[1, 7, 8, 64] = torch.ops.aten.view.default(view_10, [1, 7, -1, 64])
    view_11 = np.reshape(view_10, [1, 7, -1, 64])
    # permute_6: f32[1, 8, 7, 64] = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3])
    permute_6 = np.transpose(view_11, [0, 2, 1, 3])

    unsqueeze_3 = np.expand_dims(mul, 1)

    unsqueeze_4 = np.expand_dims(mul_1, 1)

    mul_4 = np.multiply(permute_2, unsqueeze_3)

    slice_4 = permute_2[:, :, :, 0:32]

    slice_5 = permute_2[:, :, :, 32:]

    neg = np.negative(slice_5)
    # cat_1: f32[1, 32, 7, 64] = torch.ops.aten.cat.default([neg, slice_4], -1)
    cat_1 = np.concatenate([neg, slice_4], -1)

    mul_5 = np.multiply(cat_1, unsqueeze_4)
    # add_1: f32[1, 32, 7, 64] = torch.ops.aten.add.Tensor(mul_4, mul_5)
    add_1 = np.add(mul_4, mul_5)

    mul_6 = np.multiply(permute_4, unsqueeze_3)

    slice_6 = permute_4[:, :, :, 0:32]

    slice_7 = permute_4[:, :, :, 32:]

    neg_1 = np.negative(slice_7)
    # cat_2: f32[1, 8, 7, 64] = torch.ops.aten.cat.default([neg_1, slice_6], -1)
    cat_2 = np.concatenate([neg_1, slice_6], -1)

    mul_7 = np.multiply(cat_2, unsqueeze_4)
    # add_2: f32[1, 8, 7, 64] = torch.ops.aten.add.Tensor(mul_6, mul_7)
    add_2 = np.add(mul_6, mul_7)

    index_put = np.copy(arg4_1)
    # index_put_np_0 = np.expand_dims(arg2_1, axis=(0, 1, 3))
    # np.put_along_axis(index_put, index_put_np_0, add_2, 2)
    index_put[:, :, arg2_1] = add_2

    index_put_1 = np.copy(arg11_1)
    # index_put_1_np_0 = np.expand_dims(arg2_1, axis=(0, 1, 3))
    # np.put_along_axis(index_put_1, index_put_1_np_0, permute_6, 2)
    index_put_1[:, :, arg2_1] = permute_6

    # slice_12 = index_put[0:]
    ## slice_13: f32[1, 8, 16, 64] = torch.ops.aten.slice.Tensor(slice_12, 1, 0, 9223372036854775807)
    # slice_13 = slice_12[0:, 0:]
    slice_13 = index_put[0:, 0:]
    # unsqueeze_6: f32[1, 8, 1, 16, 64] = torch.ops.aten.unsqueeze.default(slice_13, 2)
    unsqueeze_6 = np.expand_dims(slice_13, 2)
    # slice_14: f32[1, 8, 1, 16, 64] = torch.ops.aten.slice.Tensor(unsqueeze_6, 3, 0, 9223372036854775807)
    # slice_14 = unsqueeze_6[:, :, :, 0:]
    # # slice_15: f32[1, 8, 1, 16, 64] = torch.ops.aten.slice.Tensor(slice_14, 4, 0, 9223372036854775807)
    # slice_15 = slice_14[:, :, :, :, 0:]
    slice_15 = unsqueeze_6[:, :, :, 0:, 0:]
    # expand_4: f32[1, 8, 4, 16, 64] = torch.ops.aten.expand.default(slice_15, [1, 8, 4, 16, 64])
    expand_4 = np.broadcast_to(slice_15, [1, 8, 4, 16, 64])
    # clone: f32[1, 8, 4, 16, 64] = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format)
    clone = np.copy(
        expand_4,
    )
    # view_12: f32[1, 32, 16, 64] = torch.ops.aten.view.default(clone, [1, 32, 16, 64])
    view_12 = np.reshape(clone, [1, 32, 16, 64])

    # slice_20 = index_put_1[0:]
    # # slice_21: f32[1, 8, 16, 64] = torch.ops.aten.slice.Tensor(slice_20, 1, 0, 9223372036854775807)
    # slice_21 = slice_20[:, 0:]
    slice_21 = index_put_1[0:, 0:]
    # unsqueeze_8: f32[1, 8, 1, 16, 64] = torch.ops.aten.unsqueeze.default(slice_21, 2)
    unsqueeze_8 = np.expand_dims(slice_21, 2)
    # slice_22: f32[1, 8, 1, 16, 64] = torch.ops.aten.slice.Tensor(unsqueeze_8, 3, 0, 9223372036854775807)
    # slice_22 = unsqueeze_8[:, :, :, 0:]
    # # slice_23: f32[1, 8, 1, 16, 64] = torch.ops.aten.slice.Tensor(slice_22, 4, 0, 9223372036854775807)
    # slice_23 = slice_22[:, :, :, :, 0:]
    slice_23 = unsqueeze_8[:, :, :, 0:, 0:]
    # expand_6: f32[1, 8, 4, 16, 64] = torch.ops.aten.expand.default(slice_23, [1, 8, 4, 16, 64])
    expand_6 = np.broadcast_to(slice_23, [1, 8, 4, 16, 64])
    # clone_1: f32[1, 8, 4, 16, 64] = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format)
    clone_1 = np.copy(
        expand_6,
    )
    # view_13: f32[1, 32, 16, 64] = torch.ops.aten.view.default(clone_1, [1, 32, 16, 64])
    view_13 = np.reshape(clone_1, [1, 32, 16, 64])

    permute_7 = np.transpose(view_12, [0, 1, 3, 2])
    # expand_7: f32[1, 32, 7, 64] = torch.ops.aten.expand.default(add_1, [1, 32, 7, 64])
    expand_7 = np.broadcast_to(add_1, [1, 32, 7, 64])
    # view_14: f32[32, 7, 64] = torch.ops.aten.view.default(expand_7, [32, 7, 64])
    view_14 = np.reshape(expand_7, [32, 7, 64])
    # expand_8: f32[1, 32, 64, 16] = torch.ops.aten.expand.default(permute_7, [1, 32, 64, 16])
    expand_8 = np.broadcast_to(permute_7, [1, 32, 64, 16])
    # view_15: f32[32, 64, 16] = torch.ops.aten.view.default(expand_8, [32, 64, 16])
    view_15 = np.reshape(expand_8, [32, 64, 16])
    # bmm_1: f32[32, 7, 16] = torch.ops.aten.bmm.default(view_14, view_15)
    bmm_1 = np.matmul(view_14, view_15)
    # view_16: f32[1, 32, 7, 16] = torch.ops.aten.view.default(bmm_1, [1, 32, 7, 16])
    view_16 = np.reshape(bmm_1, [1, 32, 7, 16])
    # mul_8: f32[1, 32, 7, 16] = torch.ops.aten.mul.Tensor(view_16, 0.125)
    mul_8 = np.multiply(view_16, 0.125)

    # slice_24 = arg5_1[0:]
    # # slice_25: f32[1, 1, 7, 16] = torch.ops.aten.slice.Tensor(slice_24, 1, 0, 9223372036854775807)
    # slice_25 = slice_24[:, 0:]
    # # slice_26: f32[1, 1, 7, 16] = torch.ops.aten.slice.Tensor(slice_25, 2, 0, 9223372036854775807)
    # slice_26 = slice_25[:, :, 0:]
    slice_26 = arg5_1[0:, 0:, 0:]

    add_3 = np.add(mul_8, slice_26)

    add_3_np_0 = np.max(add_3, axis=-1, keepdims=True)
    add_3_np_1 = np.exp(np.subtract(add_3, add_3_np_0))
    _softmax = np.divide(add_3_np_1, np.sum(add_3_np_1, axis=-1, keepdims=True))

    clone_2 = np.copy(
        _softmax,
    )

    expand_9 = np.broadcast_to(clone_2, [1, 32, 7, 16])
    # view_17: f32[32, 7, 16] = torch.ops.aten.view.default(expand_9, [32, 7, 16])
    view_17 = np.reshape(expand_9, [32, 7, 16])
    # expand_10: f32[1, 32, 16, 64] = torch.ops.aten.expand.default(view_13, [1, 32, 16, 64])
    expand_10 = np.broadcast_to(view_13, [1, 32, 16, 64])
    # view_18: f32[32, 16, 64] = torch.ops.aten.view.default(expand_10, [32, 16, 64])
    view_18 = np.reshape(expand_10, [32, 16, 64])
    # bmm_2: f32[32, 7, 64] = torch.ops.aten.bmm.default(view_17, view_18)
    bmm_2 = np.matmul(view_17, view_18)
    # view_19: f32[1, 32, 7, 64] = torch.ops.aten.view.default(bmm_2, [1, 32, 7, 64])
    view_19 = np.reshape(bmm_2, [1, 32, 7, 64])

    permute_8 = np.transpose(view_19, [0, 2, 1, 3])
    # clone_3: f32[1, 7, 32, 64] = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format)
    clone_3 = np.copy(
        permute_8,
    )

    view_20 = np.reshape(clone_3, [1, 7, -1])

    permute_9 = np.transpose(arg12_1, [1, 0])
    # view_21: f32[7, 2048] = torch.ops.aten.view.default(view_20, [7, 2048])
    view_21 = np.reshape(view_20, [7, 2048])
    # mm_3: f32[7, 2048] = torch.ops.aten.mm.default(view_21, permute_9)
    mm_3 = np.matmul(view_21, permute_9)
    # view_22: f32[1, 7, 2048] = torch.ops.aten.view.default(mm_3, [1, 7, 2048])
    view_22 = np.reshape(mm_3, [1, 7, 2048])

    add_4 = np.add(embedding, view_22)

    pow_2 = np.power(add_4, 2)
    # mean_1: f32[1, 7, 1] = torch.ops.aten.mean.dim(pow_2, [-1], True)
    pow_2_np_0 = np.sum(pow_2, axis=(-1,), keepdims=True)
    mean_1 = np.divide(pow_2_np_0, pow_2.shape[-1])

    add_5 = np.add(mean_1, 1e-05)
    # rsqrt_1: f32[1, 7, 1] = torch.ops.aten.rsqrt.default(add_5)
    add_5_np_0 = np.sqrt(add_5)
    rsqrt_1 = np.divide(1, add_5_np_0)
    # mul_9: f32[1, 7, 2048] = torch.ops.aten.mul.Tensor(add_4, rsqrt_1)
    mul_9 = np.multiply(add_4, rsqrt_1)

    mul_10 = np.multiply(arg13_1, mul_9)

    permute_10 = np.transpose(arg14_1, [1, 0])
    # view_23: f32[7, 2048] = torch.ops.aten.view.default(mul_10, [7, 2048])
    view_23 = np.reshape(mul_10, [7, 2048])
    # mm_4: f32[7, 8192] = torch.ops.aten.mm.default(view_23, permute_10)
    mm_4 = np.matmul(view_23, permute_10)
    # view_24: f32[1, 7, 8192] = torch.ops.aten.view.default(mm_4, [1, 7, 8192])
    view_24 = np.reshape(mm_4, [1, 7, 8192])
    # sigmoid: f32[1, 7, 8192] = torch.ops.aten.sigmoid.default(view_24)
    view_24_np_0 = np.negative(view_24)
    view_24_np_1 = np.exp(view_24_np_0)
    view_24_np_2 = np.add(1, view_24_np_1)
    sigmoid = np.divide(1, view_24_np_2)
    # mul_11: f32[1, 7, 8192] = torch.ops.aten.mul.Tensor(view_24, sigmoid)
    mul_11 = np.multiply(view_24, sigmoid)
    # permute_11: f32[2048, 8192] = torch.ops.aten.permute.default(arg15_1, [1, 0])
    permute_11 = np.transpose(arg15_1, [1, 0])
    # view_25: f32[7, 2048] = torch.ops.aten.view.default(mul_10, [7, 2048])
    view_25 = np.reshape(mul_10, [7, 2048])
    # mm_5: f32[7, 8192] = torch.ops.aten.mm.default(view_25, permute_11)
    mm_5 = np.matmul(view_25, permute_11)
    # view_26: f32[1, 7, 8192] = torch.ops.aten.view.default(mm_5, [1, 7, 8192])
    view_26 = np.reshape(mm_5, [1, 7, 8192])
    # mul_12: f32[1, 7, 8192] = torch.ops.aten.mul.Tensor(mul_11, view_26)
    mul_12 = np.multiply(mul_11, view_26)
    # permute_12: f32[8192, 2048] = torch.ops.aten.permute.default(arg16_1, [1, 0])
    permute_12 = np.transpose(arg16_1, [1, 0])
    # view_27: f32[7, 8192] = torch.ops.aten.view.default(mul_12, [7, 8192])
    view_27 = np.reshape(mul_12, [7, 8192])
    # mm_6: f32[7, 2048] = torch.ops.aten.mm.default(view_27, permute_12)
    mm_6 = np.matmul(view_27, permute_12)
    # view_28: f32[1, 7, 2048] = torch.ops.aten.view.default(mm_6, [1, 7, 2048])
    view_28 = np.reshape(mm_6, [1, 7, 2048])

    add_6 = np.add(add_4, view_28)

    pow_3 = np.power(add_6, 2)
    # mean_2: f32[1, 7, 1] = torch.ops.aten.mean.dim(pow_3, [-1], True)
    pow_3_np_0 = np.sum(pow_3, axis=(-1,), keepdims=True)
    mean_2 = np.divide(pow_3_np_0, pow_3.shape[-1])

    add_7 = np.add(mean_2, 1e-05)
    # rsqrt_2: f32[1, 7, 1] = torch.ops.aten.rsqrt.default(add_7)
    add_7_np_0 = np.sqrt(add_7)
    rsqrt_2 = np.divide(1, add_7_np_0)
    # mul_13: f32[1, 7, 2048] = torch.ops.aten.mul.Tensor(add_6, rsqrt_2)
    mul_13 = np.multiply(add_6, rsqrt_2)

    mul_14 = np.multiply(arg17_1, mul_13)

    # slice_27 = mul_14[0:]
    ## slice_28: f32[1, 1, 2048] = torch.ops.aten.slice.Tensor(slice_27, 1, -1, 9223372036854775807)
    # slice_28 = slice_27[:, -1:]
    ## slice_29: f32[1, 1, 2048] = torch.ops.aten.slice.Tensor(slice_28, 2, 0, 9223372036854775807)
    # slice_29 = slice_28[:, :, 0:]
    slice_29 = mul_14[:, mul_14.shape[1] - 1 :, 0:]
    # permute_13: f32[2048, 128256] = torch.ops.aten.permute.default(arg1_1, [1, 0])
    permute_13 = np.transpose(arg1_1, [1, 0])
    # view_29: f32[1, 2048] = torch.ops.aten.view.default(slice_29, [1, 2048])
    view_29 = np.reshape(slice_29, [1, 2048])
    # mm_7: f32[1, 128256] = torch.ops.aten.mm.default(view_29, permute_13)
    mm_7 = np.matmul(view_29, permute_13)
    # view_30: f32[1, 1, 128256] = torch.ops.aten.view.default(mm_7, [1, 1, 128256])
    view_30 = np.reshape(mm_7, [1, 1, 128256])

    # copy__1: f32[1, 8, 16, 64] = torch.ops.aten.copy_.default(arg11_1, index_put_1)
    return (index_put, index_put_1, view_30)

    # arg0_1: i64[1, 7]


# Input specs, values and descriptive names
# N.B.: names and descriptions are generated by LLM and can be incorrect
# N.B.: random values are also generated by LLM and can be incorrect

# Token IDs - typically small integers representing vocabulary tokens
# Using vocabulary range of 50,000
arg0_1 = np.random.randint(0, 50000, (1, 7)).astype(np.int32)

# Embedding table - typically initialized with small normal distribution values
# Mean 0, std 0.02 is common for transformer embeddings
arg1_1 = np.random.normal(0, 0.02, (128256, 2048)).astype(np.float32)

# Position indices for KV cache - sequential indices within context window
arg2_1 = np.arange(7).astype(np.int32)  # [0,1,2,3,4,5,6]

# Rotary position indices - typically match the sequence positions
arg3_1 = np.expand_dims(np.arange(7), 0).astype(np.int32)  # [[0,1,2,3,4,5,6]]

# KV cache - typically starts as zeros or small values
# Using small normal distribution to simulate partially filled cache
arg4_1 = np.random.normal(0, 0.02, (1, 8, 16, 64)).astype(np.float32)

# Attention mask - typically has large negative values for masked positions
# and zeros for valid positions
mask = np.zeros((1, 1, 7, 16), dtype=np.float32)
# Create causal mask (upper triangular part is masked)
for i in range(7):
    for j in range(16):
        if j > i:  # This creates a causal mask (can't attend to future tokens)
            mask[0, 0, i, j] = -10000.0
arg5_1 = mask

# Rotary base frequencies - typically logarithmically spaced values
# theta_i = 10000^(-2i/d)
dim = 64
inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2).astype(np.float32) / dim))
arg6_1 = inv_freq.astype(np.float32)

# Layer norm weights - typically initialized as ones
arg7_1 = np.ones(2048, dtype=np.float32)

# Weight matrices for projections - Xavier/Glorot initialization
# std = sqrt(2.0 / (in_features + out_features))

# Query weights
std = np.sqrt(2.0 / (2048 + 2048))
arg8_1 = np.random.normal(0, std, (2048, 2048)).astype(np.float32)

# Key weights
std = np.sqrt(2.0 / (512 + 2048))
arg9_1 = np.random.normal(0, std, (512, 2048)).astype(np.float32)

# Value weights
std = np.sqrt(2.0 / (512 + 2048))
arg10_1 = np.random.normal(0, std, (512, 2048)).astype(np.float32)

# Value cache - similar to key cache
arg11_1 = np.random.normal(0, 0.02, (1, 8, 16, 64)).astype(np.float32)

# Attention output projection weights
std = np.sqrt(2.0 / (2048 + 2048))
arg12_1 = np.random.normal(0, std, (2048, 2048)).astype(np.float32)

# Second layer norm weights - ones
arg13_1 = np.ones(2048, dtype=np.float32)

# MLP up-projection weights
std = np.sqrt(2.0 / (8192 + 2048))
arg14_1 = np.random.normal(0, std, (8192, 2048)).astype(np.float32)

# MLP gate weights
std = np.sqrt(2.0 / (8192 + 2048))
arg15_1 = np.random.normal(0, std, (8192, 2048)).astype(np.float32)

# MLP down-projection weights
std = np.sqrt(2.0 / (2048 + 8192))
arg16_1 = np.random.normal(0, std, (2048, 8192)).astype(np.float32)

# Final layer norm weights - ones
arg17_1 = np.ones(2048, dtype=np.float32)


TOKEN_IDS_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 7], default=(1, 7)),
    dtype_spec=CommonTypes.INTS,
    description="Input token IDs",
    default_value=arg0_1,
)

EMBEDDING_TABLE_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[128256, 2048], default=(128256, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Token embedding table",
    default_value=arg1_1,
)

POSITION_INDICES_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[7], default=(7,)),
    dtype_spec=CommonTypes.INTS,
    description="Position indices for KV cache updating",
    default_value=arg2_1,
)

ROTARY_POSITION_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 7], default=(1, 7)),
    dtype_spec=CommonTypes.INTS,
    description="Position indices for rotary embeddings",
    default_value=arg3_1,
)

KEY_CACHE_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 8, 16, 64], default=(1, 8, 16, 64)),
    dtype_spec=CommonTypes.FLOATS,
    description="Key cache tensor",
    default_value=arg4_1,
)

ATTENTION_MASK_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 1, 7, 16], default=(1, 1, 7, 16)),
    dtype_spec=CommonTypes.FLOATS,
    description="Attention mask tensor",
    default_value=arg5_1,
)

ROTARY_BASE_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[32], default=(32,)),
    dtype_spec=CommonTypes.FLOATS,
    description="Rotary embedding base frequencies",
    default_value=arg6_1,
)

LN1_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[2048], default=(2048,)),
    dtype_spec=CommonTypes.FLOATS,
    description="First layer normalization weights",
    default_value=arg7_1,
)

QUERY_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[2048, 2048], default=(2048, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Query projection weights",
    default_value=arg8_1,
)

KEY_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[512, 2048], default=(512, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Key projection weights",
    default_value=arg9_1,
)

VALUE_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[512, 2048], default=(512, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Value projection weights",
    default_value=arg10_1,
)

VALUE_CACHE_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[1, 8, 16, 64], default=(1, 8, 16, 64)),
    dtype_spec=CommonTypes.FLOATS,
    description="Value cache tensor",
    default_value=arg11_1,
)

ATTN_OUTPUT_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[2048, 2048], default=(2048, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="Attention output projection weights",
    default_value=arg12_1,
)

LN2_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[2048], default=(2048,)),
    dtype_spec=CommonTypes.FLOATS,
    description="Second layer normalization weights",
    default_value=arg13_1,
)

MLP_UP_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[8192, 2048], default=(8192, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="MLP up-projection weights",
    default_value=arg14_1,
)

MLP_GATE_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[8192, 2048], default=(8192, 2048)),
    dtype_spec=CommonTypes.FLOATS,
    description="MLP gate weights for SwiGLU activation",
    default_value=arg15_1,
)

MLP_DOWN_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[2048, 8192], default=(2048, 8192)),
    dtype_spec=CommonTypes.FLOATS,
    description="MLP down-projection weights",
    default_value=arg16_1,
)

FINAL_LN_WEIGHT_SPEC = TensorInputSpec(
    shape_spec=ShapeSpec(dims=[2048], default=(2048,)),
    dtype_spec=CommonTypes.FLOATS,
    description="Final layer normalization weights",
    default_value=arg17_1,
)

# Define the kernel spec
kernel_specs = [
    KernelSpec(
        function=nkipy_kernel_func,
        inputs=[
            TOKEN_IDS_SPEC,
            EMBEDDING_TABLE_SPEC,
            POSITION_INDICES_SPEC,
            ROTARY_POSITION_SPEC,
            KEY_CACHE_SPEC,
            ATTENTION_MASK_SPEC,
            ROTARY_BASE_SPEC,
            LN1_WEIGHT_SPEC,
            QUERY_WEIGHT_SPEC,
            KEY_WEIGHT_SPEC,
            VALUE_WEIGHT_SPEC,
            VALUE_CACHE_SPEC,
            ATTN_OUTPUT_WEIGHT_SPEC,
            LN2_WEIGHT_SPEC,
            MLP_UP_WEIGHT_SPEC,
            MLP_GATE_WEIGHT_SPEC,
            MLP_DOWN_WEIGHT_SPEC,
            FINAL_LN_WEIGHT_SPEC,
        ],
        description="Complete transformer decoder block with self-attention, rotary position embeddings, and SwiGLU MLP",
        is_pure_numpy=True,
    )
]
