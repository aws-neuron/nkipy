# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.activation import SiluAndMul


def custom_silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x_reshaped = x.view(-1, x.shape[-1])
    s = x_reshaped[:, :d] * F.sigmoid(x_reshaped[:, :d])
    result = s * x_reshaped[:, d:]
    return result.view(*x.shape[:-1], d)


SiluAndMul.forward_oot = custom_silu_and_mul
