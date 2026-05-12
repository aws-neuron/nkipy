# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape transformation operations:
reshape, transpose, expand_dims, concatenate, split, copy, repeat
"""

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# Primitive transform ops
# -----------------------------------------------------------------------------
reshape = Op("reshape")
transpose = Op("transpose")
expand_dims = Op("expand_dims")
concatenate = Op("concatenate")
split = Op("split")
copy = Op("copy")
repeat = Op("repeat")
broadcast_to = Op("broadcast_to")
astype = Op("astype")
squeeze = Op("squeeze")
pad = Op("pad")
swapaxes = Op("swapaxes")
stack = Op("stack")
diff = Op("diff")
flip = Op("flip")
tile = Op("tile")
roll = Op("roll")

# -----------------------------------------------------------------------------
# copyto (deprecated for HLO due to in-place semantics)
# -----------------------------------------------------------------------------
copyto = Op("copyto")
