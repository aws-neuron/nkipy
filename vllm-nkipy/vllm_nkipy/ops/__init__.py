# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# Adapted from vLLM-Ascend (https://github.com/vllm-project/vllm-ascend/blob/cadfa5ddc162a4e4ac0886524fa6400ea1dcd72d/vllm_ascend/ops/__init__.py)

import torch

import vllm_nkipy.ops.activation  # noqa
import vllm_nkipy.ops.moe.blockwise_moe  # noqa
import vllm_nkipy.ops.moe.router  # noqa
import vllm_nkipy.ops.moe.simple_moe  # noqa
import vllm_nkipy.ops.moe.utils  # noqa
import vllm_nkipy.ops.rotary_embedding  # noqa

# import vllm_nkipy.ops.sampler  # noqa


class dummyFusionOp:
    default = None

    def __init__(self, name=""):
        self.name = name


# To avoid import error
def register_dummy_fusion_op() -> None:
    torch.ops._C.silu_and_mul = dummyFusionOp(name="silu_and_mul")
    torch.ops._C.rms_norm = dummyFusionOp(name="rms_norm")
    torch.ops._C.fused_add_rms_norm = dummyFusionOp(name="fused_add_rms_norm")
    torch.ops._C.static_scaled_fp8_quant = dummyFusionOp(name="static_scaled_fp8_quant")
    torch.ops._C.dynamic_scaled_fp8_quant = dummyFusionOp(
        name="dynamic_scaled_fp8_quant"
    )
    torch.ops._C.dynamic_per_token_scaled_fp8_quant = dummyFusionOp(
        name="dynamic_per_token_scaled_fp8_quant"
    )
    torch.ops._C.rms_norm_static_fp8_quant = dummyFusionOp(
        name="rms_norm_static_fp8_quant"
    )
    torch.ops._C.fused_add_rms_norm_static_fp8_quant = dummyFusionOp(
        name="fused_add_rms_norm_static_fp8_quant"
    )
    torch.ops._C.rms_norm_dynamic_per_token_quant = dummyFusionOp(
        name="rms_norm_dynamic_per_token_quant"
    )
