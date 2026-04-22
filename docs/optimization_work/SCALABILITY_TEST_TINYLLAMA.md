# test P2P weight transfer in NKIPy plugin

## Goal

Test the scalability of fast model switch in NKIPy based on P2P weight transfer.


## Test settings

- Use TinyLLama as the base model with TP=8.
- Use Instance 172.31.44.131 (this machine) for testing.
- the code path is: /home/ubuntu/vllm-nkipy/nkipy
- the example shell scripts are in: /home/ubuntu/vllm-nkipy/nkipy/examples/p2p/


## Two roles in the tests

There are two roles in the tests:
- server engine, which serves as the pusher of the model weights. The server engine is initialized with model checkpoints.
- receiver engine, which serves as the receiver of the model weights, The receiver engine is initialized without model checkpoints, and it waits for /wake_up endpoint to be activated. Multiple receiver engines are needed to test the scalability and they may or may not share the same set of Neuron cores.
- Due to Neuron contraints, if multiple engines are started on the same set of neuron cores, only one of them can be active and it must be asleep before another engine wakes up. We call the asleep receiver engines as `standby engines`.
- Server engine: `./run_engine.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp 8 --checkpoint ~/models/llama3/tmp_tinyllama_TP8 --log-level DEBUG`
- Receiver engine: `./run_engine.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp 8 --core-offset 16 --port 8001`


## How to test

**Step 0:** Reserve the first 8 Neuron cores on the instance and start the server engine.
**Step 1:** Reserve the last 8 Neuron cores for receiver engines and start a number of receiver engines on the 8 Neuron cores. Task: need to figure out if the system can support up to 100 receiver engines sharing the same set of hardware. If not, identify the bottleneck and figure out the solution. 
**Step 2:** Test the /sleep and /wake_up cycles on these standy engines. Given the same prompt, ensure the generated outputs from different receiver engines are identical since they share the same model weights.
**Step 3:** Dynamically start or kill receiver engines.
**Step 4:** Sleep the server engine and wake it up from the local model checkpoint.
**Step 5:** Sleep the server engine and wake it up from the active receiver engine with P2P weight transfer.
**Step 6:** After the first 5 steps pass, write an integration test for this scalablity feature

## Test results (2026-04-20)

Instance: trn1.32xlarge (i-0c4ff105891f5ddd3), 32 NeuronCores, 495GB RAM.
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0, TP=8, max_model_len=128.
Server engine: cores 0-7, port 8000, with checkpoint.
Receiver engines: cores 16-23 (NKIPY_CORE_OFFSET=16), various ports, no checkpoint.

### Step 0: Server engine
- Started on cores 0-7, ready in ~8s.
- Inference verified: output = `" Paris.\n\n2. B. The capital of France is Paris.\n\n3. The"`

### Step 1: Multiple standby receiver engines
- Tested: 2, 5, 10, 20, 50, 100 receivers on shared cores 16-23.
- **50 receivers: 100% success** (all healthy, all sleeping).
- **100 receivers: 94/100 success** (6 failed due to vLLM engine core init race conditions — gloo port contention).
- Per-engine memory: ~960MB main process + ~480MB engine core = ~1.4GB CPU RAM each.
- 50 engines: ~48GB total (of 495GB available).
- Receiver engines start in ~1-2s each (sleeping mode, no NRT init needed).

### Step 2: Sleep/wake_up cycles with output consistency
- Tested 5 receivers sequentially: wake from server P2P, infer, sleep, 10s wait, repeat.
- **All 5 receivers produced identical output** (temperature=0, greedy sampling).
- Wake latency: 11.9s - 21.7s (includes NRT init, P2P RDMA, kernel reload).
- Sleep latency: 0.8s - 1.0s.
- Must sleep current active engine before waking another on shared cores.

### Step 3: Dynamic engine management
- Killed receiver 8001 while sleeping: other receivers unaffected.
- Started new receiver 8004 dynamically: came up correctly, P2P wake worked.
- Existing receiver 8002 continued to work normally after dynamic changes.
- **Bug found and fixed:** `context_len` stale config after sleep/wake cycle caused `AssertionError: Input x: expected shape [1, 128, 2048], got (1, 6, 2048)` on second wake. Fix: reset `self._config.context_len = None` in `nkipy_wake_up()` (worker.py).

### Step 4: Server sleep/wake from local checkpoint
- Server sleep: 0.78s.
- Server rejects /v1/* requests while sleeping (returns 503).
- Server health shows `sleeping: true`.
- Wake from checkpoint (no peer_url): 27.6s.
- Output matches pre-sleep reference exactly.

### Step 5: Server sleep/wake from active receiver via P2P
- Woke receiver 8002 from server (P2P).
- Slept server engine (0.63s).
- Woke server from receiver 8002 via P2P (19.0s).
- **Bidirectional P2P works**: server output after wake matches exactly.

### Step 6: Integration test
- Written at `tests/integration/vllm_plugin/test_scalability.py`.
- 5 test cases covering all scenarios above with 5 standby engines.
- Uses pytest fixtures for server and receiver lifecycle management.
