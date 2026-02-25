# SPDX-License-Identifier: Apache-2.0
import logging
import os
import subprocess
from typing import Any, Dict

import numpy as np
import pytest
import torch
from nkipy.core.compile import _set_build_dir
from offline_integration_test import vllm_integ_test

logger = logging.getLogger(__name__)

# Test configurations
BATCH_SIZE_CONFIGS = [
    {
        "title": "gpt_oss_batch1",
        "max_batch_size": 1,
    },
]

# Common parameters that don't change across tests
BASE_CONFIG = {
    "block_size": 32,
    "enable_prefix_caching": False,
    "n_positions": 10240,
    "dtype": "bfloat16",
    # XXX: using model with 128 attention heads to allow launching 128 workers
    "model_name_or_path": "meta-llama/Llama-3.1-405B",
    "tokenizer": "openai/gpt-oss-120b",
    "override_neuron_config": {
        "enable_bucketing": False,
    },
    "top_k": 1,
}


@pytest.fixture(scope="function", autouse=True)
def init_per_function():
    # always clear cache for test, as fast to compile and vscode has bug to reload env var
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.mark.parametrize("config", BATCH_SIZE_CONFIGS)
def test_openai_gpt_oss_batch_variations(config: Dict[str, Any]):
    """
    Offline vLLM inference test against open_llama_3b with different batch sizes
    """
    # os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"

    model_name = "openai/gpt-oss-120b"
    tp_degree = 8
    ep_degree = 16
    num_workers = tp_degree * ep_degree

    test_config = {**BASE_CONFIG, **config}

    max_batch_size = test_config["max_batch_size"]
    max_model_len = test_config["n_positions"]
    build_dir = f"/tmp/build/gpt-oss-120b-EP{ep_degree}-TP{tp_degree}-BS{max_batch_size}-SEQ{max_model_len}"
    _set_build_dir(build_dir)
    if os.environ.get("NEURONPY_NOT_CLEAR_BUILD_CACHE") != "1":
        subprocess.run(f"rm -rf {build_dir}", shell=True)

    os.environ["MODEL_NAME"] = model_name
    os.environ["MODEL_CHECKPOINT"] = f"/shared/ziyangx/gpt-oss-120b-bf16-moe-fp8-TP{tp_degree}"
    vllm_integ_test(num_workers=num_workers, **test_config)
