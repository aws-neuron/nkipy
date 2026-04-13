# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test: vLLM+NKIPy must match standalone NKIPy on Neuron.

Runs Qwen3-30B-A3B with greedy decoding through both standalone NKIPy
(via torchrun) and vLLM with the NKIPy plugin, then asserts exact token match.
"""

import glob
import json
import os
import socket
import subprocess
import tempfile

# Select only the nkipy plugin when vllm-neuron is also installed
os.environ["VLLM_PLUGINS"] = "nkipy"

import pytest

_HAS_NEURON = len(glob.glob("/dev/neuron*")) > 0
_QWEN3_CHECKPOINT = os.path.expanduser(
    "~/zhuangw/nkipy/examples/models/qwen3/tmp_Qwen3-30b-a3b"
)
_QWEN3_EXAMPLE_DIR = os.path.expanduser(
    "~/vllm-nkipy/nkipy/examples/models/qwen3"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not (_HAS_NEURON and os.path.isdir(_QWEN3_CHECKPOINT)),
    reason="Requires Neuron devices and Qwen3 checkpoint",
)
def test_nkipy_vllm_matches_standalone_qwen3():
    """Qwen3-30B-A3B: vLLM+NKIPy must match standalone NKIPy exactly."""
    model_name = "Qwen/Qwen3-30B-A3B"
    prompt = "The capital of France is"
    max_tokens = 20
    tp = 8

    # --- Standalone NKIPy reference via torchrun ---
    # The qwen3 example uses non-relative imports (from config import Config),
    # so we run from its directory.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                     dir=_QWEN3_EXAMPLE_DIR) as f:
        f.write(f"""
import json, os, sys, torch
import torch.distributed as dist

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "1"

dist.init_process_group()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())

from qwen3 import Qwen3Model
from config import get_config
from transformers import AutoTokenizer
from safetensors.torch import load_file

tokenizer = AutoTokenizer.from_pretrained("{model_name}")
input_ids = tokenizer("{prompt}", return_tensors="np")["input_ids"]
config = get_config("{model_name}", input_ids.shape[1], {max_tokens})

shard = os.path.join("{_QWEN3_CHECKPOINT}", f"shard_{{dist.get_rank()}}.safetensors")
weights = load_file(shard, device="cpu")
model = Qwen3Model(weights, config)

# Warmup
for i, _ in enumerate(model.generate(input_ids)):
    if i == 1: break

tokens = []
for tok in model.generate(input_ids):
    tokens.append(int(tok[0, 0]))
    if len(tokens) >= {max_tokens}: break

if dist.get_rank() == 0:
    with open("{f.name}.out", "w") as out:
        json.dump(tokens, out)
""")
        script_path = f.name

    env = os.environ.copy()
    env["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"
    with socket.socket() as s:
        s.bind(("", 0))
        master_port = str(s.getsockname()[1])
    result = subprocess.run(
        ["torchrun", "--nproc-per-node", str(tp),
         "--master-port", master_port, script_path],
        capture_output=True, text=True, timeout=600, env=env,
        cwd=_QWEN3_EXAMPLE_DIR,
    )
    out_path = script_path + ".out"
    try:
        if result.returncode != 0:
            pytest.fail(
                f"Standalone NKIPy failed:\nSTDOUT:\n{result.stdout[-2000:]}"
                f"\nSTDERR:\n{result.stderr[-2000:]}"
            )
        with open(out_path) as fout:
            ref_tokens = json.load(fout)
    finally:
        for p in (script_path, out_path):
            if os.path.exists(p):
                os.unlink(p)

    # --- vLLM + NKIPy plugin ---
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["NKIPY_CHECKPOINT"] = _QWEN3_CHECKPOINT

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name, max_num_seqs=1, max_model_len=128,
        tensor_parallel_size=tp, enforce_eager=True, dtype="bfloat16",
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate([prompt], sampling_params)
    vllm_tokens = list(outputs[0].outputs[0].token_ids)
    vllm_text = outputs[0].outputs[0].text
    del llm

    print(f"Standalone tokens: {ref_tokens}")
    print(f"vLLM      tokens: {vllm_tokens}")
    print(f"vLLM      text:   {vllm_text!r}")

    assert ref_tokens == vllm_tokens, (
        f"Token mismatch!\n  Standalone: {ref_tokens}\n  vLLM:       {vllm_tokens}"
    )
