# Test Catalog

Use `../.venv/bin/python -m pytest` from `nkipy_serving/` for scoped runs. Add a unique
`--junitxml=/tmp/<name>.xml` and
redirect stdout/stderr to a unique `/tmp/<name>.log` for each command.

## Scopes

| Scope | Coverage | Verified warm/cached time |
|---|---|---:|
| `unit` | Tests not marked `integration` | ~32s |
| `integration` | Integration tests without a model-specific device marker | ~27s |
| `device-qwen3` | Qwen3 dense TP8 serving and reload weights | ~4m total |
| `device-qwen3-moe` | Qwen3 MoE TP4 | ~12m |
| `device-gpt-oss` | GPT-OSS TP8 | ~26m |
| `device-ep` | GPT-OSS TP8+EP16, LNC=1 | ~11m |
| `device-dsv4` | DeepSeek-V4 TP8+EP8+R1, ADP=8, 4k buckets, LNC=1 | ~1h52m with compilation during the latest verified run |
| `device` | All device groups, sequentially | Model-cache dependent; budget at least 3h |
| `all` | Unit, non-device integration, and all device groups with zero skips | Model-cache dependent; budget at least 3h |

Cold timing depends on global and build-directory NEFF cache state. A fully cold DSV4 R1 4k run can
take longer than the verified 1h52m run, so use the committed 7200-second startup/worker windows.

## Commands

### unit (default)

Filtering at collection time avoids counting intentionally excluded integration tests as skips.

```bash
../.venv/bin/python -m pytest -m "not integration" -v --tb=short
```

### integration

```bash
../.venv/bin/python -m pytest --run-integration -v --tb=short \
  -m "integration and not device_gpt_oss and not device_qwen3_dense and not device_qwen3_moe and not device_ep and not device_dsv4"
```

### device-qwen3

Run both files as separate groups, matching the complete runner. Do not export DSV4 checkpoint
overrides into these commands.

```bash
env NKIPY_SERVING_QWEN3_DENSE_READY_TIMEOUT_S=3600 \
  ../.venv/bin/python -m pytest --run-integration --run-device-qwen3-dense \
  tests/test_tp8_serving.py -v --tb=short

env NKIPY_SERVING_QWEN3_DENSE_READY_TIMEOUT_S=3600 \
  ../.venv/bin/python -m pytest --run-integration --run-device-qwen3-dense \
  tests/test_tp8_reload_weights.py -v --tb=short
```

### device-qwen3-moe

```bash
env NKIPY_SERVING_QWEN3_MOE_READY_TIMEOUT_S=3600 \
  ../.venv/bin/python -m pytest --run-integration --run-device-qwen3-moe \
  tests/test_qwen3_moe_tp4_device.py -v --tb=short
```

### device-gpt-oss

```bash
env NKIPY_SERVING_GPT_OSS_READY_TIMEOUT_S=3600 \
  ../.venv/bin/python -m pytest --run-integration --run-device-gpt-oss \
  tests/test_gpt_oss_tp8_device.py -v --tb=short
```

### device-ep

```bash
env NEURON_LOGICAL_NC_CONFIG=1 \
  NKIPY_SERVING_GPT_OSS_READY_TIMEOUT_S=3600 \
  ../.venv/bin/python -m pytest --run-integration --run-device-ep \
  tests/test_gpt_oss_tp8_ep16_device.py -v --tb=short
```

### device-dsv4

Recover `DSV4_HF_MODEL_ID`, `DSV4_TOKENIZER_MODEL_ID`, and `DSV4_PREPARED_WEIGHT_DIR` from exact
paths already present in the active agent session. Validate them before running:

```bash
test -d "$DSV4_HF_MODEL_ID"
test -d "$DSV4_TOKENIZER_MODEL_ID"
test -d "$DSV4_PREPARED_WEIGHT_DIR"
```

Run the full-model server tests first:

```bash
env NEURON_LOGICAL_NC_CONFIG=1 \
  NKIPY_SERVING_HF_MODEL_ID="$DSV4_HF_MODEL_ID" \
  NKIPY_SERVING_TOKENIZER_MODEL_ID="$DSV4_TOKENIZER_MODEL_ID" \
  NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR="$DSV4_PREPARED_WEIGHT_DIR" \
  NKIPY_SERVING_DSV4_PREPARED_WEIGHT_LOCAL_DIR=/tmp/dsv4_prepared_smoke \
  NKIPY_SERVING_COLLECTIVE_LOAD_TIMEOUT_S=7200 \
  NKIPY_SERVING_TP_WORKER_TIMEOUT_S=7200 \
  NKIPY_SERVING_SCHEDULER_READY_TIMEOUT_S=7200 \
  NKIPY_SERVING_DSV4_READY_TIMEOUT_S=7200 \
  NKIPY_SERVING_DSV4_REQUEST_TIMEOUT_S=3600 \
  NKIPY_SERVING_DSV4_DEVICE_CONFIG=tests/runtime.tp8_ep8_r1.deepseek_v4.multi_bucket_4k.test.json \
  ../.venv/bin/python -m pytest --run-integration --run-device-dsv4 \
  tests/test_deepseek_v4_device.py -v --tb=short
```

After that process and its workers have fully exited, run bucket state-write
coverage on one visible core:

```bash
env NEURON_LOGICAL_NC_CONFIG=1 \
  NEURON_RT_VISIBLE_CORES=0 \
  ../.venv/bin/python -m pytest --run-integration --run-device-dsv4 \
  tests/test_dsv4_writeswa_bucket_device.py -v --tb=short
```

### device

Run the five device scopes above in the same order as `scripts/run_all_tests.sh`. Keep each file or
model family as a separate reported group and clean up stale test workers between groups.

### all

Export the session-derived DSV4 variables and use the canonical runner:

```bash
PYTHON_BIN=../.venv/bin/python bash scripts/run_all_tests.sh
```

Append `--clean` only when explicitly requested.

## Markers and files

| Marker | Required flag | Files |
|---|---|---|
| `integration` only | `--run-integration` | Non-device integration tests, including NKI sampling accuracy |
| `device_qwen3_dense` | `--run-device-qwen3-dense` | `test_tp8_serving.py`, `test_tp8_reload_weights.py` |
| `device_qwen3_moe` | `--run-device-qwen3-moe` | `test_qwen3_moe_tp4_device.py` |
| `device_gpt_oss` | `--run-device-gpt-oss` | `test_gpt_oss_tp8_device.py` |
| `device_ep` | `--run-device-ep` | `test_gpt_oss_tp8_ep16_device.py` |
| `device_dsv4` | `--run-device-dsv4` | `test_deepseek_v4_device.py`, `test_dsv4_writeswa_bucket_device.py` |

## Cache directories

| Directory | Primary owner |
|---|---|
| `/tmp/build` | General runtime and integration tests |
| `/tmp/build_tp8` | Qwen3 dense TP8 |
| `/tmp/build_tp4` | Qwen3 MoE TP4 |
| `/tmp/build_tp8_ep16` | GPT-OSS TP8+EP16 |
| `/tmp/build_gpt_oss_bench` | GPT-OSS benchmark/device artifacts |
| `/tmp/build_dsv4_smoke` | DeepSeek-V4 device artifacts |

For a complete clean run, rely on `scripts/run_all_tests.sh --clean` so the cache list stays aligned
with the executable test matrix.
