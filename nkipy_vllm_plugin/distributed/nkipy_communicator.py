# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed import ProcessGroup
from vllm.config import get_current_vllm_config
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger

USE_RAY = parallel_config = (
    get_current_vllm_config().parallel_config.distributed_executor_backend == "ray"
)

logger = init_logger(__name__)


class NKIPyCommunicator(DeviceCommunicatorBase):

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: Optional[torch.device] = None,
        device_group: Optional[ProcessGroup] = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)

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
