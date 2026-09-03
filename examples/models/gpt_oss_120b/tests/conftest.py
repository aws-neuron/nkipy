import os
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
from nkipy.core.compile import _get_build_dir, _set_build_dir
from nkipy.runtime import device_kernel

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parallel_state import initialize_model_parallel

# Test configuration
TEST_TP_FOR_SHAPE = 8

@pytest.fixture(scope="function", autouse=True)
def init_per_function():
    # always clear cache for test, because it is fast to compile. In additional, vscode has bug to reload env var
    subprocess.run(f"rm -rf {_get_build_dir()}", shell=True)
    np.random.seed(42)
    torch.manual_seed(42)
    device_kernel._LOADED_KERNELS.clear()

@pytest.fixture(scope="session", autouse=True)
def init_per_session():
    os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"
    worker = os.environ.get('PYTEST_XDIST_WORKER')
    if worker:
        worker_id = int(worker.replace('gw', ''))
    else:
        worker_id = 0
    _set_build_dir(f"/tmp/build/test/worker{worker_id}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_available_port(start_port=61239 + worker_id))
    os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    # TODO: currently only support single core test
    dist.init_process_group(rank=0, world_size=1)
    os.environ['NEURON_RT_VISIBLE_CORES'] = str(worker_id)
    initialize_model_parallel(tp_size=1, prefill_ep_size=1)

def find_available_port(start_port=61239, max_tries=100):
    """Find an available port starting from start_port.
    Increments by 128 to leave room for parallel workers."""
    for i in range(max_tries):
        port = start_port + (i * 128)  # Jump by 128 to avoid worker collisions
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find available port after {max_tries} tries")
