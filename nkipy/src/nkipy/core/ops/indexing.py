# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexing operations: where, take, take_along_axis, put_along_axis,
static_slice, dynamic_update_slice, scatter_strided
"""

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# Primitive indexing ops
# -----------------------------------------------------------------------------
where = Op("where")
take = Op("take")
take_along_axis = Op("take_along_axis")
scatter_along_axis = Op("scatter_along_axis")
put_along_axis = Op("put_along_axis")
static_slice = Op("static_slice")
dynamic_update_slice = Op("dynamic_update_slice")
scatter_strided = Op("scatter_strided")
