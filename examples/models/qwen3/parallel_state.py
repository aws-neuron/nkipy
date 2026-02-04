import os

import torch
import torch.distributed as dist


def initialize_model_parallel():
    dist.init_process_group()
    torch.set_num_threads(128 // dist.get_world_size())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())
