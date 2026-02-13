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
    """Convert input to column matrix for efficient convolution via matmul.

    Args:
        input_padded: Padded input tensor of shape (batch, in_channels, height, width)
        kernel_h, kernel_w: Kernel dimensions
        stride_h, stride_w: Stride values
        out_h, out_w: Output spatial dimensions

    Returns:
        Column matrix of shape (batch, in_channels * kernel_h * kernel_w, out_h * out_w)
    """
    batch_size, in_channels, _, _ = input_padded.shape

    # Use stride_tricks to create a view of sliding windows
    # Shape: (batch, in_channels, out_h, out_w, kernel_h, kernel_w)
    shape = (batch_size, in_channels, out_h, out_w, kernel_h, kernel_w)
    strides = (
        input_padded.strides[0],  # batch
        input_padded.strides[1],  # channel
        input_padded.strides[2] * stride_h,  # out_h (strided)
        input_padded.strides[3] * stride_w,  # out_w (strided)
        input_padded.strides[2],  # kernel_h
        input_padded.strides[3],  # kernel_w
    )

    windows = np.lib.stride_tricks.as_strided(
        input_padded, shape=shape, strides=strides
    )

    # Reshape to (batch, in_channels * kernel_h * kernel_w, out_h * out_w)
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
    """Convert input to column matrix for efficient 3D convolution via matmul.

    Args:
        input_padded: Padded input tensor of shape
                      (batch, in_channels, depth, height, width)
        kernel_d, kernel_h, kernel_w: Kernel dimensions
        stride_d, stride_h, stride_w: Stride values
        out_d, out_h, out_w: Output spatial dimensions

    Returns:
        Column matrix of shape
        (batch, in_channels * kernel_d * kernel_h * kernel_w, out_d * out_h * out_w)
    """
    batch_size, in_channels, _, _, _ = input_padded.shape

    # Use stride_tricks to create a view of sliding windows
    # Shape: (batch, in_channels, out_d, out_h, out_w, kernel_d, kernel_h, kernel_w)
    shape = (batch_size, in_channels, out_d, out_h, out_w, kernel_d, kernel_h, kernel_w)
    strides = (
        input_padded.strides[0],  # batch
        input_padded.strides[1],  # channel
        input_padded.strides[2] * stride_d,  # out_d (strided)
        input_padded.strides[3] * stride_h,  # out_h (strided)
        input_padded.strides[4] * stride_w,  # out_w (strided)
        input_padded.strides[2],  # kernel_d
        input_padded.strides[3],  # kernel_h
        input_padded.strides[4],  # kernel_w
    )

    windows = np.lib.stride_tricks.as_strided(
        input_padded, shape=shape, strides=strides
    )

    # (batch, in_channels * kernel_d * kernel_h * kernel_w, out_d * out_h * out_w)
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
    """2D Convolution operation (CPU) using im2col approach.

    Args:
        input: Input tensor of shape (batch, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride for convolution (int or tuple)
        padding: Padding for input (int or tuple)
        dilation: Dilation for kernel (int or tuple) - NOT IMPLEMENTED
        groups: Number of groups for grouped convolution - NOT IMPLEMENTED
        out: Output tensor (unused)
        dtype: Output dtype (unused)

    Returns:
        Output tensor of shape (batch, out_channels, out_height, out_width)
    """
    # Normalize parameters
    stride = _normalize_tuple_2d(stride, "stride")
    padding = _normalize_tuple_2d(padding, "padding")
    dilation = _normalize_tuple_2d(dilation, "dilation")

    # Check for unsupported features
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

    # Get dimensions
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

    # Pad input if necessary
    if pad_h > 0 or pad_w > 0:
        input_padded = np.pad(
            input,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        input_padded = np.ascontiguousarray(input)

    # Ensure input is contiguous for stride_tricks
    if not input_padded.flags["C_CONTIGUOUS"]:
        input_padded = np.ascontiguousarray(input_padded)

    # im2col: convert input to column matrix
    # Shape: (batch, in_channels * kernel_h * kernel_w, out_height * out_width)
    col = _im2col_2d(
        input_padded, kernel_h, kernel_w, stride_h, stride_w, out_height, out_width
    )

    # Reshape weight for matrix multiplication
    # Shape: (out_channels, in_channels * kernel_h * kernel_w)
    weight_reshaped = weight.reshape(out_channels, -1)

    # Perform convolution as matrix multiplication
    # weight_reshaped: (out_channels, in_channels * kernel_h * kernel_w)
    # col: (batch, in_channels * kernel_h * kernel_w, out_h * out_w)
    # We want: (batch, out_channels, out_h * out_w)
    # Use matmul: weight_reshaped @ col[b] for each batch
    output = np.matmul(weight_reshaped, col)

    # Reshape to output shape
    output = output.reshape(batch_size, out_channels, out_height, out_width)

    # Add bias if provided
    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)

    return output


@conv2d.impl("hlo")
def _conv2d_hlo(
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
    """2D Convolution operation for HLO."""
    from nkipy.core.backend.hlo import broadcast_to_shape_hlo, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(input, NKIPyTensorRef):
        input = input.backend_tensor
    if isinstance(weight, NKIPyTensorRef):
        weight = weight.backend_tensor

    stride = _normalize_tuple_2d(stride, "stride")
    dilation = _normalize_tuple_2d(dilation, "dilation")
    padding_tuple = _normalize_tuple_2d(padding, "padding")

    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape

    out_height = (
        in_height + 2 * padding_tuple[0] - dilation[0] * (kernel_height - 1) - 1
    ) // stride[0] + 1
    out_width = (
        in_width + 2 * padding_tuple[1] - dilation[1] * (kernel_width - 1) - 1
    ) // stride[1] + 1

    output_shape = (batch_size, out_channels, out_height, out_width)

    padding_config = [
        (padding_tuple[0], padding_tuple[0]),
        (padding_tuple[1], padding_tuple[1]),
    ]

    result_tensor = ctx.build_op(
        "convolution",
        [input, weight],
        output_shape,
        input.dtype,
        {
            "window_strides": list(stride),
            "padding": padding_config,
            "lhs_dilation": [1, 1],
            "rhs_dilation": list(dilation),
            "feature_group_count": groups,
            "batch_group_count": 1,
            "input_batch_dimension": 0,
            "input_feature_dimension": 1,
            "input_spatial_dimensions": [2, 3],
            "kernel_output_feature_dimension": 0,
            "kernel_input_feature_dimension": 1,
            "kernel_spatial_dimensions": [2, 3],
            "output_batch_dimension": 0,
            "output_feature_dimension": 1,
            "output_spatial_dimensions": [2, 3],
        },
    )

    if bias is not None:
        if isinstance(bias, NKIPyTensorRef):
            bias = bias.backend_tensor

        bias_reshaped = ctx.build_op(
            "reshape", [bias], (1, out_channels, 1, 1), bias.dtype
        )
        bias_broadcast = broadcast_to_shape_hlo(ctx, bias_reshaped, output_shape)
        result_tensor = ctx.build_op(
            "add", [result_tensor, bias_broadcast], output_shape, input.dtype
        )

    return NKIPyTensorRef(result_tensor)


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
    """3D Convolution operation (CPU) using im2col approach.

    Args:
        input: Input tensor of shape (batch, in_channels, depth, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_d,
                                        kernel_h, kernel_w)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride for convolution (int or tuple)
        padding: Padding for input (int or tuple)
        dilation: Dilation for kernel (int or tuple) - NOT IMPLEMENTED
        groups: Number of groups for grouped convolution - NOT IMPLEMENTED
        out: Output tensor (unused)
        dtype: Output dtype (unused)

    Returns:
        Output tensor of shape (batch, out_channels, out_depth, out_height, out_width)
    """
    # Normalize parameters
    stride = _normalize_tuple_3d(stride, "stride")
    padding = _normalize_tuple_3d(padding, "padding")
    dilation = _normalize_tuple_3d(dilation, "dilation")

    # Check for unsupported features
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

    # Get dimensions
    batch_size, in_channels, in_depth, in_height, in_width = input.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape

    # Calculate output dimensions
    out_depth = (in_depth + 2 * pad_d - kernel_d) // stride_d + 1
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

    # Pad input if necessary
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        input_padded = np.pad(
            input,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        input_padded = np.ascontiguousarray(input)

    # Ensure input is contiguous for stride_tricks
    if not input_padded.flags["C_CONTIGUOUS"]:
        input_padded = np.ascontiguousarray(input_padded)

    # im2col: convert input to column matrix
    # (batch, in_channels * kernel_d * kernel_h * kernel_w,
    # out_depth * out_height * out_width)
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

    # Reshape weight for matrix multiplication
    # Shape: (out_channels, in_channels * kernel_d * kernel_h * kernel_w)
    weight_reshaped = weight.reshape(out_channels, -1)

    # Perform convolution as matrix multiplication
    # weight_reshaped: (out_channels, in_channels * kernel_d * kernel_h * kernel_w)
    # col: (batch, in_channels * kernel_d * kernel_h * kernel_w, out_d * out_h * out_w)
    # We want: (batch, out_channels, out_d * out_h * out_w)
    output = np.matmul(weight_reshaped, col)

    # Reshape to output shape
    output = output.reshape(batch_size, out_channels, out_depth, out_height, out_width)

    # Add bias if provided
    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1, 1)

    return output


@conv3d.impl("hlo")
def _conv3d_hlo(
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
    """3D Convolution operation for HLO."""
    from nkipy.core.backend.hlo import broadcast_to_shape_hlo, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(input, NKIPyTensorRef):
        input = input.backend_tensor
    if isinstance(weight, NKIPyTensorRef):
        weight = weight.backend_tensor

    stride = _normalize_tuple_3d(stride, "stride")
    dilation = _normalize_tuple_3d(dilation, "dilation")
    padding_tuple = _normalize_tuple_3d(padding, "padding")

    batch_size, in_channels, in_depth, in_height, in_width = input.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape

    out_depth = (
        in_depth + 2 * padding_tuple[0] - dilation[0] * (kernel_depth - 1) - 1
    ) // stride[0] + 1
    out_height = (
        in_height + 2 * padding_tuple[1] - dilation[1] * (kernel_height - 1) - 1
    ) // stride[1] + 1
    out_width = (
        in_width + 2 * padding_tuple[2] - dilation[2] * (kernel_width - 1) - 1
    ) // stride[2] + 1

    output_shape = (batch_size, out_channels, out_depth, out_height, out_width)

    padding_config = [
        (padding_tuple[0], padding_tuple[0]),
        (padding_tuple[1], padding_tuple[1]),
        (padding_tuple[2], padding_tuple[2]),
    ]

    result_tensor = ctx.build_op(
        "convolution",
        [input, weight],
        output_shape,
        input.dtype,
        {
            "window_strides": list(stride),
            "padding": padding_config,
            "lhs_dilation": [1, 1, 1],
            "rhs_dilation": list(dilation),
            "feature_group_count": groups,
            "batch_group_count": 1,
            "input_batch_dimension": 0,
            "input_feature_dimension": 1,
            "input_spatial_dimensions": [2, 3, 4],
            "kernel_output_feature_dimension": 0,
            "kernel_input_feature_dimension": 1,
            "kernel_spatial_dimensions": [2, 3, 4],
            "output_batch_dimension": 0,
            "output_feature_dimension": 1,
            "output_spatial_dimensions": [2, 3, 4],
        },
    )

    if bias is not None:
        if isinstance(bias, NKIPyTensorRef):
            bias = bias.backend_tensor

        bias_reshaped = ctx.build_op(
            "reshape", [bias], (1, out_channels, 1, 1, 1), bias.dtype
        )
        bias_broadcast = broadcast_to_shape_hlo(ctx, bias_reshaped, output_shape)
        result_tensor = ctx.build_op(
            "add", [result_tensor, bias_broadcast], output_shape, input.dtype
        )

    return NKIPyTensorRef(result_tensor)
