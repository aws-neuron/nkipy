# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Flash Paged Attention kernels with variable-length sequence inputs.
"""

import neuronxcc.nki.language as nl

B_P_SIZE = nl.tile_size.pmax
B_FMAX_SIZE = nl.tile_size.gemm_moving_fmax

NEG_INF = -9984.0  # Magic number to replace -inf similar to what Tensorizer uses
