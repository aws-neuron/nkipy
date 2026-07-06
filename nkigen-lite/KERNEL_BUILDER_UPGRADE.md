# Kernel Builder upgrade opportunities

`nki` was bumped from `0.3.0+23928721754.g18aa1271` to `0.6.0b1` (editable,
`../../private-nki-staging`) in `nkipy/pyproject.toml`. That upgrade required
one compatibility fix (`fori_loop` now defaults to `unroll=False`; nkigen-lite
now passes `unroll=True` explicitly — see `emit_to_kb.py:839,854`) but also
brought new `nisa`/kernel_builder primitives nkigen-lite currently emulates in
software. This doc tracks those — check items off as they're addressed.

Full API diff method: compared `dir(nki.compiler.kernel_builder)` and
`dir(nki.compiler.kernel_builder.isa)` plus per-function `inspect.signature`
between the two versions. See conversation history (2026-07-05) for the raw
diffs; only the entries below have a live caller in nkigen-lite or the qwen3
examples.

## Blocked

### 1. [ ] Replace the software top-k scan with native `nisa.topk` — blocked on `neuronx-cc`

`nkigen-lite/src/nkigen_lite/tensor_ir/passes/basic/direct_lower.py:786-900`
(`_topk_scan`, `_load_topk_data`, `_emit_topk_op`) + the `TOPK_FREE_MAX = 16384`
chunking constant at line 783.

**Current state**: nkigen-lite's `topk` (`tensor_ir/ir.py:406`) lowers to a
hand-rolled hardware scan because nki_ir had no native top-k primitive:
`ceil(k/8)` rounds of `max8` (next 8 largest) + `match_replace8` (record
indices, mask taken values to `-inf`), with intermediate folds round-tripped
through an HBM scratch buffer (`_topk_scan`, no on-chip sub-tile column write
available). For `F > TOPK_FREE_MAX` it splits into chunks, runs a local top-k
per chunk with rebased indices, then does a second merge scan over the
candidates — ~150 lines across `direct_lower.py` and `ir.py`.

**What's new**: `nisa.topk(val_dst, idx_dst, src, n, name=None)` — a single
GpSIMD-engine instruction that finds the `n` largest values and indices from a
source tile directly (`nisa_apis.py:2486`, doc: *"Find the K largest values
and their indices from a source tile using GpSIMD Engine"*).

**Why it matters**: `topk` is on the sampling hot path for both qwen3 examples
— `examples/models/qwen3/kernels/sampling.py:129,137` (greedy decode, `k=1`)
and `examples/models/qwen3/kernels/transformer_layer.py:100` (top-k sampling).
Every generated token pays for this. Swapping in the native op should both cut
latency (one instruction vs. a multi-round scan with HBM round-trips) and
delete the most complex lowering code in `direct_lower.py`.

**Blocker (found 2026-07-05)**: hand-built a minimal kernel_builder kernel
calling `nisa.topk` and compiled it directly (bypassing nkigen-lite) — the
pinned `neuronx-cc` (`2.24.8799.0+6f62ff7c`, registry) does not implement the
`Topk` BIR opcode at all:

```
Unknown opcode Topk
UNREACHABLE executed at .../walrus/ir/lib/IR/Instruction.cpp:941!
```

This is a hard compiler-backend gap, not a usage error — the `nisa.topk`
Python/MLIR frontend exists in `private-nki-staging` 0.6.0b1, but the paired
NCC lowering isn't in the version nkipy-oss is pinned to. Checked whether the
latest available `neuronx-cc` (`2.25.3371.0+f524f7f8`, pip index) fixes this —
inconclusive: that wheel is cp311-only (nkipy-oss's venv is cp310) and setting
up a clean standalone cp311 probe environment hit unrelated tooling issues
before reaching a real answer. Even if 2.25 does support it, bumping
`neuronx-cc` is a separate, larger decision (it's nkipy's registry-pinned
backend for everything, not scoped to this one op) and shouldn't be done as a
side effect of adding topk.

**Before starting** (once unblocked): check `nisa.topk`'s actual hardware
constraints — max `n`, max source free-dim width (does it need the
`TOPK_FREE_MAX` chunk-and-merge wrapper at all, or does the hardware op handle
wide `F` natively?), tie-break behavior on equal values (must match the
current `max8`/`match_replace8` semantics that `nkigen-lite/tests/tensor_ir/
test_*.py` numerically pin), and dtype support for `val_dst`/`idx_dst`. Add a
nki_ir-level op (`nisa.topk` is one instruction, no loop) and switch
`_emit_topk_op` to emit it directly when constraints allow, keeping the
chunked fallback only if the hardware op has a narrower `F` limit than
`TOPK_FREE_MAX`.

**Next step**: get a working `neuronx-cc` >= the version that ships `Topk`
support onto this box (matching nkipy-oss's cp310 venv), confirm the opcode
compiles, then proceed with the plan above.

## Surveyed, not worth chasing yet

New kernel_builder/`isa` surface with no current caller in nkigen-lite or the
qwen3 examples — re-check if a future kernel needs them:

- `nisa.core_barrier(data, cores, engine=None)` — multi-NeuronCore
  synchronization barrier, usable now (NeuronCore-v3+, matches nkigen-lite's
  `trn2` target). Only relevant if nkigen-lite grows LNC>1 kernels that need
  explicit cross-core sync beyond the existing collectives.
- `nisa.activate2(dst, src, imm0, imm1, op, op0, op1, relu_param=None, ...)` —
  fused tensor-scalar activation with two immediates + optional reduce, Scalar
  engine. **Gated to NeuronCore-v4 and newer** (`python/nki/isa/_activation.py:329`
  docstring) — nkigen-lite currently targets `trn2` (NeuronCore-v3), so this is
  not usable at all yet, not just uncalled. Re-check when/if the target moves
  to v4.
- `nisa.all_gather_v(dsts, srcs, replica_group, metadata_tensor, ...)` —
  variable-length all-gather with per-destination send/receive counts and
  displacements (no hardware-version gate found). Meaningfully different from
  plain `all_gather`: built for ragged, per-destination-sized transfers.
  Worth a closer look for qwen3's MoE token dispatch/combine (variable
  tokens routed per expert) if that ever needs to move off the current
  fixed-shape `all_gather`/`all_to_all` pattern — but current MoE usage is
  fixed-shape, so no immediate driver.
- `nisa.inline_asm_bytes(asm_bytes, sync_type, engine, latency_ns, ...)` — raw
  hardware instruction escape hatch. Only needed if a NISA op nkigen-lite wants
  has no builder-level wrapper yet.

Confirmed backward-compatible, no action needed: `engine=None` on
`tensor_copy`/`tensor_tensor_arith`/`tensor_scalar_arith`/the collective ops
means "auto-assign" and resolves to the same default engine as the old
explicit `Engine.Vector`/`Engine.Gpsimd` defaults (verified against
`nisa_apis.py` source); the new `dma_qos` kwarg on the collective ops is
optional and unused by nkigen-lite; `select_reduce`'s `on_false` has no
default in either version and nkigen-lite's emitter already always supplies
it.
