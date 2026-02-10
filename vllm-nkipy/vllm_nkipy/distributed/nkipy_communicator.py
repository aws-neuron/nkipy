# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# Adapted from vLLM (https://github.com/vllm-project/vllm)

from typing import Optional

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class NKIpyCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: Optional[torch.device] = None,
        device_group: Optional[ProcessGroup] = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return funcol.all_reduce(input_, "sum", self.device_group)

    def all_gather(self, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            # Convert negative dim to positive.
            dim += input.dim()
        # output_tensor = custom_all_gather(
        output_tensor = funcol.all_gather_tensor(
            input,
            dim,
            self.device_group,
        )
        return output_tensor
