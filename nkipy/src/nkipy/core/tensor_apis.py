# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward compatible support for tensor APIs"""

from nkipy.core.ops import conv2d, conv3d, full, rms_norm, softmax, topk, zeros

__all__ = ["topk", "rms_norm", "softmax", "conv2d", "conv3d", "zeros", "full"]
