# nkigen-lite Backend — Status Report

*As of 2026-06-30 · branch `nkigen-lite` · 530 tests in `nkigen-lite/tests/`*

## Executive summary

The nkigen-lite backend is **functionally complete for the two Qwen3 reference
models** and **numerically verified against HuggingFace/HLO on real Trainium
hardware**. Both examples produce correct results end-to-end. The remaining gap
is **performance**, not correctness.

| Model | Compiles | Runs e2e | Numerically correct | Notes |
|-------|:--------:|:--------:|:-------------------:|-------|
| **Qwen3-30B-A3B** (generative, MoE, TP=4) | yes | yes | yes — matches HLO | 0.38 tok/s vs HLO 38.1 (~100x); MoE loop still dominates |
| **Qwen3-Embedding-0.6B** (single-core) | yes | yes | yes — min cosine 0.9997 vs HF | default backend; ~20x slower than HLO after lowering fixes (see Performance) |

## What works

- **Full op coverage** for both models: ~50 NISA ops in `emit_to_kb`, and 12
  high-level tensor_ir ops lower through `direct_lower` (matmul, reduce,
  transpose, reshape, slice, concat, broadcast, iota, topk, gather/scatter
  rows, gather_along_axis), plus elementwise fusion and collectives
  (all_reduce / all_gather / reduce_scatter / all_to_all).
- **Fused MoE transformer layer** (prefill + decode) compiles and runs within
  SBUF.
- **TP=4 collectives** verified correct in the full 30B run.
- **Verification done on-device**, not just unit tests: greedy_sampling, fused
  sampling+embedding, wide elementwise, and tiled topk all checked against
  numpy; both models checked against reference implementations.

## Performance (HLO vs nkigen-lite)

nkigen-lite is correct but still slower than HLO. **Full numbers, root-cause
analysis, progress log, and backlog live in [PERFORMANCE.md](PERFORMANCE.md)** —
this section is just the headline.

**Qwen3-Embedding-0.6B** (input (1, 128), LNC=2, trn2):

| Backend | Mean latency | Throughput | vs HLO |
|---------|-------------:|-----------:|-------:|
| HLO | 8.57 ms | 116.7 inf/s | 1.0x |
| nkigen-lite — original baseline | 1007.20 ms | 0.99 inf/s | 118x |
| nkigen-lite — current | 174.07 ms | 5.74 inf/s | 20x |

The gap is **per-instruction / serialization overhead, not bandwidth**: one
fused layer originally expanded 396 tensor_ir ops → ~82.6k nki_ir ops (208x),
mostly tiny per-batch-element DMAs from data-movement lowering that iterated
leading dims one element at a time. Three fixes so far — broadcast → collapsed
`(L, B, T)`, elementwise and concat/slice → collapse-leading-dims-onto-partition
— cut that 118x → 20x (5.8x faster). The next lever is keeping intermediates in
SBUF across consecutive ops. See PERFORMANCE.md for the per-op breakdown and
`examples/models/qwen3_embedding/profile_layer.py` to reproduce.

**Qwen3-30B-A3B** (generative, TP=4): HLO ~44 tok/s vs nkigen-lite ~0.23 tok/s
(~190x slower) — same systemic cause; not yet re-profiled since the broadcast
fix.

(To reproduce both backends, BACKEND in qwen3_embedding/model.py is now
env-overridable via QWEN3_BACKEND, default nkigen-lite.)

## What was fixed to get here (this work, 3 commits)

1. **SBUF OOMs in the fused layer** (`e6b6de0`):
   - reshape -> zero-copy HBM view (was copying element-by-element; ~8.5x
     op-count reduction)
   - emit_to_kb scratch tiles (`tensor_scalar_arith` f32 upcast,
     `affine_select`, `scalar_const`) now released instead of leaking
   - elementwise innermost free-dim capped to per-partition budget (vocab-wide
     rows were 150 KB tiles)
2. **topk free-dim tiling** (`e6b6de0`): `max8` hardware-caps free dim at
   16384; sharded vocab is 37984. Added chunk-and-merge tiling.
3. **Example portability** (`8c806be`): sliced prefill mask; RoPE caches +
   decode causal mask passed as runtime kernel inputs (avoids non-uniform
   constant overflow); backend-aware neff I/O naming (`_alias_key`,
   `_map_outputs`).

## Known limitations / open items

- **Performance** (biggest gap): 30B generative runs at ~0.23 tok/s, ~190x
  slower than HLO. Likely causes: fully-unrolled 64-expert MoE loop, per-op HBM
  round-trips in the lowering, wide-row tiling overhead. Not yet investigated.
- **`shared_constant` data-constant path is blocked** by a neuronx-cc bug
  (`get_shared_constant` -> `dma_copy` "operand does not dominate"). Worked
  around by passing RoPE/mask caches as kernel inputs; the cleaner long-term
  fix needs the upstream bug resolved.
- **Stale-build-cache footgun**: must `rm -rf build/` after code changes — a
  stale cache once produced wrong output that looked like a numerical bug.
  Worth documenting/automating.
- **No device test for wide-topk tiling** (F>16384): verified manually but
  uncommitted; existing topk tests only cover F<=16384.
- **Runtime-integration gaps** (pre-existing, surfaced during this work):
  nkipy's `DeviceKernel` runtime can't load an nkigen-lite neff directly
  (NRT_INVALID), and neffs with collectives won't load single-core. Worked
  around in tests via `nb.compile_and_execute`.

## Suggested next actions

1. **Performance investigation** on the 30B layer (the one substantive gap).
2. Add a **wide-topk device regression test** (cheap, prevents silent
   breakage).
3. File the **`shared_constant` neuronx-cc bug** upstream.
4. Decide on **push / PR** for the `nkigen-lite` branch (3 commits unpushed).

## How to reproduce

```bash
# Qwen3-30B-A3B generative (TP=4) — needs a 4-shard checkpoint
rm -rf examples/models/qwen3/build/
QWEN3_BACKEND=nkigen-lite uv run torchrun --nproc-per-node 4 \
    examples/models/qwen3/qwen3.py -n 16 \
    --checkpoint /home/ubuntu/models/Qwen3-30B-A3B-TP4 \
    --model Qwen/Qwen3-30B-A3B "The capital of France is"
# -> "...Paris. The capital of the United Kingdom is London..."

# Qwen3-Embedding-0.6B (single process) — nkigen-lite is the default backend
cd examples/models/qwen3_embedding && rm -rf build/
uv run python example_retrieval.py --compare
# -> min cosine similarity 0.9997 vs HuggingFace, PASS

# Benchmark either backend (rm -rf build/ between backends):
QWEN3_BACKEND=nkigen-lite uv run python example_retrieval.py \
    --benchmark --num-warmup 3 --num-iterations 10
QWEN3_BACKEND=hlo uv run python example_retrieval.py \
    --benchmark --num-warmup 3 --num-iterations 10
```
