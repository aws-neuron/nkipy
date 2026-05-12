# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convolution operations: conv2d, conv3d"""

import numpy as np

from nkipy.core.ops._registry import Op


def _normalize_tuple_2d(value, name):
    """Normalize value to 2-tuple."""
    if isinstance(value, int):
        return (value, value)
    elif isinstance(value, (tuple, list)):
        if len(value) == 2:
            return tuple(value)
        else:
            raise ValueError(f"{name} must be an int or a tuple of 2 ints, got {value}")
    else:
        raise ValueError(f"{name} must be an int or a tuple of 2 ints, got {value}")


def _normalize_tuple_3d(value, name):
    """Normalize value to 3-tuple."""
    if isinstance(value, int):
        return (value, value, value)
    elif isinstance(value, (tuple, list)):
        if len(value) == 3:
            return tuple(value)
        else:
            raise ValueError(f"{name} must be an int or a tuple of 3 ints, got {value}")
    else:
        raise ValueError(f"{name} must be an int or a tuple of 3 ints, got {value}")


def _im2col_2d(input_padded, kernel_h, kernel_w, stride_h, stride_w, out_h, out_w):
    """Convert input to column matrix for efficient convolution via matmul."""
    batch_size, in_channels, _, _ = input_padded.shape

    shape = (batch_size, in_channels, out_h, out_w, kernel_h, kernel_w)
    strides = (
        input_padded.strides[0],
        input_padded.strides[1],
        input_padded.strides[2] * stride_h,
        input_padded.strides[3] * stride_w,
        input_padded.strides[2],
        input_padded.strides[3],
    )

    windows = np.lib.stride_tricks.as_strided(
        input_padded, shape=shape, strides=strides
    )

    col = windows.transpose(0, 1, 4, 5, 2, 3).reshape(
        batch_size, in_channels * kernel_h * kernel_w, out_h * out_w
    )

    return col


def _im2col_3d(
    input_padded,
    kernel_d,
    kernel_h,
    kernel_w,
    stride_d,
    stride_h,
    stride_w,
    out_d,
    out_h,
    out_w,
):
    """Convert input to column matrix for efficient 3D convolution via matmul."""
    batch_size, in_channels, _, _, _ = input_padded.shape

    shape = (batch_size, in_channels, out_d, out_h, out_w, kernel_d, kernel_h, kernel_w)
    strides = (
        input_padded.strides[0],
        input_padded.strides[1],
        input_padded.strides[2] * stride_d,
        input_padded.strides[3] * stride_h,
        input_padded.strides[4] * stride_w,
        input_padded.strides[2],
        input_padded.strides[3],
        input_padded.strides[4],
    )

    windows = np.lib.stride_tricks.as_strided(
        input_padded, shape=shape, strides=strides
    )

    col = windows.transpose(0, 1, 5, 6, 7, 2, 3, 4).reshape(
        batch_size, in_channels * kernel_d * kernel_h * kernel_w, out_d * out_h * out_w
    )

    return col


# -----------------------------------------------------------------------------
# conv2d
# -----------------------------------------------------------------------------
conv2d = Op("conv2d")


@conv2d.impl("cpu")
def _conv2d_cpu(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    out=None,
    dtype=None,
):
    """2D Convolution operation (CPU) using im2col approach."""
    stride = _normalize_tuple_2d(stride, "stride")
    padding = _normalize_tuple_2d(padding, "padding")
    dilation = _normalize_tuple_2d(dilation, "dilation")

    if dilation != (1, 1):
        raise NotImplementedError(
            f"conv2d CPU backend does not support dilation != 1, "
            f"got dilation={dilation}"
        )
    if groups != 1:
        raise NotImplementedError(
            f"conv2d CPU backend does not support groups != 1, got groups={groups}"
        )

    stride_h, stride_w = stride
    pad_h, pad_w = padding

    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

    if pad_h > 0 or pad_w > 0:
        input_padded = np.pad(
            input,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        input_padded = np.ascontiguousarray(input)

    if not input_padded.flags["C_CONTIGUOUS"]:
        input_padded = np.ascontiguousarray(input_padded)

    col = _im2col_2d(
        input_padded, kernel_h, kernel_w, stride_h, stride_w, out_height, out_width
    )

    weight_reshaped = weight.reshape(out_channels, -1)
    output = np.matmul(weight_reshaped, col)
    output = output.reshape(batch_size, out_channels, out_height, out_width)

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)

    return output


# -----------------------------------------------------------------------------
# conv3d
# -----------------------------------------------------------------------------
conv3d = Op("conv3d")


@conv3d.impl("cpu")
def _conv3d_cpu(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    out=None,
    dtype=None,
):
    """3D Convolution operation (CPU) using im2col approach."""
    stride = _normalize_tuple_3d(stride, "stride")
    padding = _normalize_tuple_3d(padding, "padding")
    dilation = _normalize_tuple_3d(dilation, "dilation")

    if dilation != (1, 1, 1):
        raise NotImplementedError(
            f"conv3d CPU backend does not support dilation != 1, "
            f"got dilation={dilation}"
        )
    if groups != 1:
        raise NotImplementedError(
            f"conv3d CPU backend does not support groups != 1, got groups={groups}"
        )

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    batch_size, in_channels, in_depth, in_height, in_width = input.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape

    out_depth = (in_depth + 2 * pad_d - kernel_d) // stride_d + 1
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        input_padded = np.pad(
            input,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        input_padded = np.ascontiguousarray(input)

    if not input_padded.flags["C_CONTIGUOUS"]:
        input_padded = np.ascontiguousarray(input_padded)

    col = _im2col_3d(
        input_padded,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        out_depth,
        out_height,
        out_width,
    )

    weight_reshaped = weight.reshape(out_channels, -1)
    output = np.matmul(weight_reshaped, col)
    output = output.reshape(batch_size, out_channels, out_depth, out_height, out_width)

    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1, 1)

    return output
