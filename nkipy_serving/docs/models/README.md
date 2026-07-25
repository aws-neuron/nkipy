# Models

Model-specific notes for the Neuron runtime.

## Supported Paths

| Model family | Model id pattern | Attention backend | Execution mode | Notes |
|---|---|---|---|---|
| Qwen3 Dense | `Qwen/Qwen3-*` without `-A<size>B` | `NKIBlockSparseFlashAttention` | All-layers-one-graph (embedding outside) | Shared prepared attention inputs + unified worker warmup |
| Qwen3 MoE | `Qwen/Qwen3-*-A<size>B*` | `NKIBlockSparseFlashAttention` | Per-layer prefill graphs | Shared prepared attention inputs + unified worker warmup |
| GPT-OSS MoE | `unsloth/gpt-oss-*` | `NKIBlockSparseFlashAttention` (required) | Full decode graph; per-layer prefill kernels | Prefill is fused per layer around the CPU MoE scheduling boundary |
| DeepSeek-V4 Flash | `deepseek-ai/DeepSeek-V4-Flash` | DeepSeek-V4 sparse MLA/SWA | Product layer fragments | TP=8, EP=8, R1 4k bucket path with FP8 MoE weights and global NEFF catalog reuse |

## Common Runtime Notes

- Worker warmup now means both compilation and one synthetic execution per startup bucket path.
- `NKIPY_SERVING_BUILD_DIR` or config field `nkipy_build_dir` can be used to isolate or reuse compile caches.
- Shared prepared attention inputs live in `nkipy_serving/attention/nki_step_inputs.py` and are reused by GPT-OSS, Qwen3 dense, and Qwen3 MoE.
- DeepSeek-V4 product fragments and support kernels use global signature catalogs in addition to per-config build-dir records, so fresh build directories can reuse already-compiled NEFFs without entering compile locks.

## Debug: Eager Executors

Each model ships a parallel `eager_executor.py` for debug. Eager executors
expose the layer body as swappable `@jit` fragments and include a pure-numpy
`forward_cpu()` reference. They are not a serving path. See
[../design.md §6](../design.md#6-eager-executors-debug-surface).

## Model Docs

- [Qwen3 Dense](./qwen3_dense.md)
- [GPT-OSS](./gpt_oss.md)
- [DeepSeek-V4](./deepseek_v4.md)
