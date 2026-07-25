# DeepSeek-V4

DeepSeek-V4 Flash is served through the Python-first `deepseek_v4` runtime path on trn2.48xlarge. The current documented and validated device path is TP=8, EP=8, R1, ADP=8 with `NEURON_LOGICAL_NC_CONFIG=1` and the 4k bucket config.

## Execution Path

- Product execution uses layer-level product fragments. Prefill MoE is concatenated with prefill attention, and decode MoE is concatenated with decode attention; prefill and decode are not merged into one NEFF.
- MoE weights are preprocessed to FP8 E4M3 for the serving path. The blockwise MoE kernel consumes the FP8 weights directly.
- Startup warmup compiles or loads all configured buckets before `/ready`, then executes synthetic first-touch forward steps for those buckets.
- `token_buckets` are the product-prefill shape boundary; live prompt lengths
  are covered by the selected bucket instead of a separate per-prompt prefill
  length configuration.
- `/health` is only liveness. `/ready` is the correct gate for accepting requests because it waits for kernel load, allocation, weight load, and warmup.

## Kernel Cache Layout

DeepSeek-V4 caches compiled NEFFs in two places:

- Product layer fragments: the build-dir canonical cache under `<build_dir>/.dsv4_product_canonical_neffs/`. The build dir comes from the `nkipy_build_dir` config field (default `/tmp/build`), suffixed by the config hash. The canonical-record root is de-ranked (the per-rank `rank_N/` segment is stripped to `<base>/product/`), so all collective ranks and successive server restarts that share the same `nkipy_build_dir` reuse the same compiled NEFFs.
- Support kernels: the global catalog `NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR`, default `/tmp/nkipy_serving_neff_catalog`.

The support catalog is namespace-scoped and covers:

- `dsv4_attention_kernels`: SWA, sparse attention, KV writes.
- `dsv4_compressor_kernels`: compressor-side kernels.
- `blockwise_moe`: blockwise MoE support kernels.
- `logits_processor`: LM-head, top-1, sampler, and logprob kernels.

On a fresh build directory with a populated support catalog, the runtime writes local canonical records that point at the global NEFFs. It should not create fresh local `.neff` files for those support kernels.

## Startup Diagnostics

Use startup profiling when measuring DSV4 readiness or cache behavior:

```bash
NEURON_LOGICAL_NC_CONFIG=1 \
NKIPY_SERVING_STARTUP_PROFILE=1 \
NKIPY_SERVING_DSV4_STAGE_PROFILE=1 \
NKIPY_SERVING_DSV4_WARMUP_TRACE=0 \
uv run pytest --run-integration --run-device-dsv4 \
  tests/test_deepseek_v4_device.py -v --tb=short
```

Ready time is mostly NEFF load/device initialization, device allocation, weight
upload, and synthetic first-touch forward execution. With populated global
catalogs, startup should not be dominated by neuronx-cc compilation or
compile-lock contention.

`/get_server_info` now includes `warmup_summary.worker_startup` from the
scheduler-ready payload and `scheduler_metrics.worker_startup` from live
scheduler metrics. The summary reports ready worker count, slowest ranks, and
the worst rank for each startup stage, so a ready-time regression can be
triaged without opening per-rank JSONL profiles first.

## Prepared Weight Production And Staging

The runtime consumes prepared rank-local weights, not the raw HF checkpoint.
The artifact flow is:

- Raw HF snapshot -> converted serving snapshot, typically named
  `DeepSeek-V4-Flash-neuron-fp8-noscale`. Routed expert MXFP4 tensors are
  emitted as no-scale Neuron FP8 E4M3, shared experts are emitted as BF16, and
  sidecar files are copied through.
- Converted serving snapshot -> prepared root, named for the runtime topology,
  typically `DeepSeek-V4-Flash-neuron-fp8-noscale-prepared-tp8-ep8-r1` for the
  R1 4k config. This contains rank-local `dense.safetensors`, per-layer
  `layer_NNN.safetensors`, and `metadata.json` files under topology
  directories such as `tp8_ep8_rep1/lane00_tp00/`.
- Prepared root -> optional local staged root. This is a local-disk copy used
  to avoid many workers reading large rank files from shared storage during
  startup.

Create the converted serving snapshot once per source checkpoint:

```bash
uv run python -m scripts.convert_dsv4_checkpoint \
  --src /path/to/DeepSeek-V4-Flash-hf \
  --dst /path/to/DeepSeek-V4-Flash-neuron-fp8-noscale \
  --workers 8
```

Create the TP8/EP8/R1 root for the 4k bucket config with matching replica and
attention-DP degrees:

```bash
uv run python -m scripts.prepare_dsv4_rank_weights \
  --src /path/to/DeepSeek-V4-Flash-neuron-fp8-noscale \
  --dst-root /path/to/DeepSeek-V4-Flash-neuron-fp8-noscale-prepared-tp8-ep8-r1 \
  --tp-degree 8 \
  --ep-degree 8 \
  --replica-degree 1 \
  --attention-dp-degree 8 \
  --all-unique-ranks \
  --jobs 8
```

Validate prepared roots before staging them into a device run:

```bash
uv run python -m scripts.validate_dsv4_prepared_weights \
  --root /path/to/DeepSeek-V4-Flash-neuron-fp8-noscale-prepared-tp8-ep8-r1 \
  --tp-degree 8 \
  --ep-degree 8 \
  --replica-degree 1 \
  --num-hidden-layers 43 \
  --num-routed-experts 256 \
  --expected-count 64
```

`--all-unique-ranks` prepares one directory for every TP rank and every
attention lane needed at runtime. With TP=8, EP=8, and R1, the prepared root has
64 unique rank directories. The expected unique rank-directory count is still
64 because R1 uses 8 attention-DP lanes rather than duplicating replica lanes.

The default check is cheap: metadata, directory layout, expected runtime
coverage, and non-empty safetensors files. Add `--check-safetensors` when the
headers should also be opened to catch corrupt files.

For large shared-storage prepared roots, stage the unique rank directories onto
local storage before launching the server:

```bash
uv run python -m scripts.stage_dsv4_prepared_weights \
  --src-root /path/to/DeepSeek-V4-Flash-neuron-fp8-noscale-prepared-tp8-ep8-r1 \
  --local-root /tmp/dsv4_prepared_dsv4_fp8_tp8_ep8_r1 \
  --jobs 8 \
  --expected-count 64
```

Set `dsv4_prepared_weight_dir` or
`NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR` to the prepared root. Set
`dsv4_prepared_weight_local_dir` or
`NKIPY_SERVING_DSV4_PREPARED_WEIGHT_LOCAL_DIR` to the staged local root when
local staging is used. For server-managed prestaging, set
`dsv4_prepared_weight_prestage=true` and tune host copy parallelism with
`dsv4_prepared_weight_prestage_workers`.

## 4k Bucket Config

The 4k bucket target is lower-concurrency R1. Use
`tests/runtime.tp8_ep8_r1.deepseek_v4.multi_bucket_4k.test.json` with:

- `replica_degree=1` and `attention_dp_degree=8`
- `max_context_len=4096`, `kv_pool_size=4096`, and `dsv4_state_size=4096`
- `request_buckets=[1,8]`
- `token_buckets=[256,1024,2048,4096]`

This path reduces request-owner state while keeping NEFF residency policy
unchanged. Future larger contexts should reduce product
scratch materialization or fuse stages; do not rely on unloading resident NEFFs
as the fit mechanism.

## Runtime Organization

DeepSeek-V4 model code is split by runtime role:

- `assembly/` builds the generation-forward surface from loaded device weights,
  constructs runtime components, and wires graph functions, logits processing,
  and NEFF-backed execution together.
- Top-level `graph_types.py`, `execution_capabilities.py`, `shapes.py`,
  `variants.py`, and `constants.py` hold shared graph metadata, pure shape
  contracts, QKV variant identifiers, and sparse-attention constants.
- `neff_graphs/` contains pure NumPy trace functions used to compile NEFF
  fragments. Registry construction is descriptor-driven in `neff_graphs/registry.py`;
  leaf graph files stay side-effect free and device-runtime agnostic.
- `neff_runtime/` owns shared runtime state, component installation,
  compiled-fragment caches, bucket resources, lifecycle helpers, and stage
  mixins that execute NEFF-backed product paths. Bucket resource dataclasses and
  kernel-cache helpers live under `neff_runtime/resources/`, QKV runtime
  families live under `neff_runtime/qkv/`, and runtime-owned attention/MoE
  execution lives under `neff_runtime/stages/attention_execution.py` and
  `neff_runtime/moe/execution.py`.
- `neff_runtime/attention_ops/` contains stateful model-owned attention,
  compressor, indexer, and QKV-variant execution helpers. The selected QKV
  variant name remains the profiling `qkv_path` and runtime cache key boundary.
- `neff_compiler.py` is the thin compile/run wrapper shared by graph fragments,
  and top-level `executor.py` is the model-facing executor entry point.

Historical `compiled`, `generation`, `product`, `sampled`, and old graph
execution shims are not kept; import the role-specific modules directly.

## Bucket-Prefill Experiment Harness

For bucket-prefill regression checks, run the standalone host experiments:

```bash
uv run pytest tests/test_dsv4_bucket_prefill_experiment.py -q
uv run python scripts/dsv4_bucket_prefill_experiment.py --json
```

These experiments cover the split between graph and kernel work:

- The graph contract is nkipy-style bucket-shaped tensor composition: the same
  bucket signature covers live lengths such as 10, 37, and 129, while valid rows
  match the live-shaped reference and padded rows stay zero.
- The guarded SWA scatter recipe redirects padding and old prefill rows to the
  guard owner, matching the device-tested compressor-state write path.
- Prefill MoE remains an NKI kernel wrapped by nkipy. The experiment documents
  the current blocker: the existing total-real-token scheduler is safe for
  packed/single-request rows, but not request-major multi-request bucket
  padding. The mask-based experiment schedule is the intended production shape.

The corresponding device-level cache-write validation remains:

```bash
NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_VISIBLE_CORES=0 \
  uv run pytest --run-integration --run-device-dsv4 \
  tests/test_dsv4_writeswa_bucket_device.py -q
```

## Testing

Device tests require a trn2.48xlarge, `NEURON_LOGICAL_NC_CONFIG=1`, a converted
serving checkpoint, tokenizer files, and a prepared TP8/EP8/R1 root. The
committed config uses portable model ids and skips unless the checkpoint,
tokenizer, and prepared-weight roots are supplied through env vars.

Use this environment shape for device smoke and 4k validation:

```bash
NEURON_LOGICAL_NC_CONFIG=1 \
NKIPY_SERVING_HF_MODEL_ID=/path/to/DeepSeek-V4-Flash-neuron-fp8-noscale \
NKIPY_SERVING_TOKENIZER_MODEL_ID=/path/to/DeepSeek-V4-Flash-neuron-fp8-noscale \
NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR=/path/to/DeepSeek-V4-Flash-neuron-fp8-noscale-prepared-tp8-ep8-r1 \
NKIPY_SERVING_DSV4_DEVICE_CONFIG=tests/runtime.tp8_ep8_r1.deepseek_v4.multi_bucket_4k.test.json \
NKIPY_SERVING_COLLECTIVE_LOAD_TIMEOUT_S=3600 \
NKIPY_SERVING_TP_WORKER_TIMEOUT_S=3600 \
NKIPY_SERVING_SCHEDULER_READY_TIMEOUT_S=7200 \
NKIPY_SERVING_DSV4_READY_TIMEOUT_S=7200 \
NKIPY_SERVING_DSV4_REQUEST_TIMEOUT_S=3600 \
uv run pytest --run-integration --run-device-dsv4 \
  tests/test_deepseek_v4_device.py -v --tb=short
```

For shared-storage prepared roots, either set
`NKIPY_SERVING_DSV4_PREPARED_WEIGHT_LOCAL_DIR` to a staged local copy or enable
server-managed prestaging. Override `NKIPY_SERVING_BUILD_DIR` when isolating a
test cache; otherwise the config's `nkipy_build_dir` selects the build root.

The test uses `/ready` as the startup gate, then checks `/get_server_info`, one
OpenAI chat completion, native `/generate` with logprobs, a long-prefill
request that lands in the larger token bucket, and batch>1 decode across the
configured request bucket.

Run the SWA bucket write device coverage with the same env prefix:

```bash
NEURON_LOGICAL_NC_CONFIG=1 \
NEURON_RT_VISIBLE_CORES=0 \
uv run pytest --run-integration --run-device-dsv4 \
  tests/test_dsv4_writeswa_bucket_device.py -q
```

Warm-cache timing depends on the local NEFF catalogs and build dir. The current
4k full device file has been observed around 20-25 minutes warm-cache; cold
runs that change cache keys can take longer. Keep the 7200s startup timeout for
full-file validation.

The multi-bucket device test intentionally does not configure per-prompt
prefill-length overrides. Short prompts such as the chat smoke request must run
under the selected token bucket without needing live-length warmup entries.

Operational gotchas:

- Do not kill worker processes with a shell command that contains the literal
  worker `spawn_main` string in its own command line; that pattern can
  self-match. Inspect holders with a split literal such as
  `ps aux | grep -F 'spawn'_main | grep -v grep`.
- Validate import and package-layout changes from the main editable checkout,
  not an auxiliary git worktree. Editable installs can keep `.pth` entries
  pointing at the main tree's `src/`, which makes worktree test results
  misleading for import-path changes.
- Kernel compile cache keys include the defining function module and qualname.
  Moving a file that defines a traceable kernel function changes its module
  path and causes a one-time cold recompile. Mixin and orchestration files are
  move-safe because they are not compiled kernel definitions.

## Basic Accuracy Plan

The accuracy target is component-first: keep cheap unit coverage for runtime
configuration, scheduler behavior, sparse-attention trace functions, and
prepared-weight validation. Full real-model semantic quality remains an
integration target, not a unit-test target.

| Component | CPU/reference source | Current coverage |
|---|---|---|
| Product config/layout | RuntimeConfig, model registry, rank layout, heterogeneous KV metadata | `tests/test_runtime_config.py`, `tests/test_http_server_unit.py`, and `tests/test_scheduler.py` check DSV4 config validation, server-info metadata, request-lane scheduling, and checkpoint restore behavior. |
| SWA/sparse attention prep | CPU oracles in `attention/deepseek_v4/kernels/` | `tests/test_dsv4_sparse_attention_fragment.py` checks paged SWA decode trace behavior against CPU oracles. |
| Prepared weights | Prepared-weight metadata and per-rank file layout | `tests/test_dsv4_prepared_weight_validator.py` checks metadata, fallback rank layout, missing files, and CLI error paths. |
| Device bucket coverage | Full TP8/EP8/R1 4k server on Trainium | `tests/test_deepseek_v4_device.py` launches the selected multi-bucket config and exercises config reporting, chat, native generation, long prefill, and batch decode. `tests/test_dsv4_writeswa_bucket_device.py` validates bucketed SWA state writes. |
| Scheduler/runtime token accuracy | Deterministic generated token ids | `tests/test_scheduler.py::test_deterministic_generation_token_ids_follow_worker_top1_sequence` verifies exact prompt, completion, and output ids through extend/decode. Full real-model semantic accuracy is still separate. |

## Current Caveats

- Real-checkpoint semantic quality, stochastic default-sampling quality, and
  broader bucket coverage should be validated independently from the unit suite.
- The global catalogs are signature-based; changing compiler flags, bucket config, model layout, or kernel signature creates a new cache entry.
