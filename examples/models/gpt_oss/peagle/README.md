# P-EAGLE Speculative Decoding for gpt-oss on Trainium

Parallel-drafting speculative decoding using [P-EAGLE](https://arxiv.org/abs/2602.01469)
for the gpt-oss model family on AWS Trainium. Generates K draft tokens in a
**single forward pass** (not K sequential passes), then verifies them against the
target in one multi-token target forward.

## Setup

``` sh
cd nkipy
uv sync --all-groups
source .venv/bin/activate
cd examples/models/gpt_oss
```

## Quickstart

### 1. Prepare weights

The target model (gpt-oss-20b) must already be prepared (see `../README.md`):

``` sh
# Target (if not already done)
python tensor_preparation.py \
    --model-name /path/to/gpt-oss-20b \
    --world-size 4 --output-dir ./tmp_gpt-oss-20b

# Drafter (P-EAGLE, replicated on every rank — small, ~3.6 GB)
python peagle/tensor_preparation.py \
    --model-name /path/to/GPT-OSS-20B-P-EAGLE \
    --output-dir ./peagle/tmp_p-eagle
```

### 2. Run speculative decoding

``` sh
TP=4
torchrun --nproc-per-node $TP peagle/speculate.py \
    --target-checkpoint ./tmp_gpt-oss-20b \
    --draft-checkpoint ./peagle/tmp_p-eagle \
    --model /path/to/gpt-oss-20b \
    --draft-model /path/to/GPT-OSS-20B-P-EAGLE \
    -n 256 -k 3 \
    "Write a Python function that implements binary search."
```

Output includes acceptance metrics:

```
Time to first token: 0.6s
Generated 256 tokens in N verify steps
Mean acceptance length: X.XX (K=7)
Decode tokens/sec: XX.XX
```

### Drafter placement (device by default)

By default the drafter forward runs **on the Neuron device**
(`kernels/drafter.py` + `drafter_model.py`). It keeps a full per-layer KV cache on
device (prefill over the prompt, then each step commits the accepted tokens and
drafts K positions attending to the whole context) — the same algorithm as the
CPU reference, so acceptance matches. Pass `--cpu-drafter` to run the CPU drafter
(`drafter_cpu.py`) instead: same math, but the layer forward runs in PyTorch on
host with a host↔device round-trip per step — the slower debug/reference path.
Measured on trn2 (TP=4, K=7, chat prompt):

| Drafter | Mean acceptance | Decode tok/s |
|---------|-----------------|--------------|
| device (default) | ~4.3 | ~50 |
| `--cpu-drafter`  | ~4.3 | ~4.8 |

The device drafter matches the CPU drafter's acceptance and is ~10× faster in
decode (fused single kernel, no per-step host↔device sync). Set `SPEC_PROFILE=1`
to print a per-step draft/verify time breakdown. Note: the first `prefill()`
compiles a full-prompt-width fused kernel, which inflates time-to-first-token on
a cold build cache.

### Choosing K (draft tokens per step)

Verify runs `K+1` candidate tokens through the target in one pass. Profiling on
trn2 (TP=4) shows its cost is **~19 ms fixed + ~6.3 ms per candidate token** — the
fixed part (a single-token forward, ~19 ms, dominated by loading the 24 layers'
weights from HBM) is amortized across all `K+1` tokens, so verify is efficient
*per token*. But acceptance saturates: this drafter plateaus around ~3.3 accepted
tokens/step, so beyond K≈3 each extra draft slot adds verify time for no extra
accepted tokens. Sweep (n=96, chat prompt):

| K | Mean acceptance | Decode tok/s |
|---|-----------------|--------------|
| 1 | 1.94 | 50.0 |
| 2 | 2.67 | 52.6 |
| **3 (default)** | **3.31** | **~60** |
| 4 | 3.31 | 49.5 |
| 7 | 4.33 | 50.4 |

**K=3 is the throughput optimum (~60–62 tok/s), beating both the base model
(~52 tok/s) and larger K.** Higher K only helps if a stronger drafter raises the
acceptance plateau. Set `-k` to override.

### Speedup over the base model

Speedup = speculative tok/s ÷ base (non-speculative) tok/s of the same target.
Measured on trn2 (TP=4, K=3, binary-search chat prompt, acceptance ~3.2):

| Configuration | Decode tok/s | Speedup |
|---|---|---|
| Base gpt-oss-20b (no speculation) | ~52 | 1.00× |
| P-EAGLE K=3, `loop` MoE (default) | 61.6 | **1.18×** |
| P-EAGLE K=3, `batched` MoE | 66.2 | **1.27×** |

The speedup is well below the ~3.2 acceptance length because verify is not free:
each step runs `K+1` tokens through the full target (~36–39 ms) plus ~13 ms of
drafting. Decode is dispatch/overhead-bound, so verifying 4 tokens costs roughly
2× a single-token step while producing ~3.2 tokens of output — hence ~1.2–1.3×.

**Speedup scales ~linearly with acceptance length, which is prompt-dependent.**
Chat-formatted, in-distribution prompts accept ~3.2 (→ ~1.2–1.3×); an
out-of-distribution / hard prompt can accept ~2.3 (→ ~1.0×, i.e. no win).
Always benchmark tok/s with a fixed, representative prompt. The biggest lever on
speedup is a stronger drafter (raises the acceptance plateau); a cheaper verify
(LNC2, or the `batched`/`dense` MoE kernels below) is the secondary lever.

### MoE kernel selection

The target's MoE feed-forward has several implementations, selected via the
`GPT_OSS_MOE_KERNEL` env var (read in `kernels/transformer_layer.py`):

| value | description | best regime |
|---|---|---|
| `loop` (default) | per-(token, expert) Python loop | reference |
| `batched` | gather top-k experts, batched GEMV | decode / verify with K ≤ 4 |
| `dense` | all experts as one dense GEMM, router-masked | verify with K ≥ 5 |
| `concat` | fuse a token's top-k experts into 2 matmuls | (≈ `batched`, no gain) |

All are numerically equivalent. End-to-end P-EAGLE tok/s (TP=4, n=160):

| K | `loop` | `batched` | `dense` |
|---|--------|-----------|---------|
| 1 | 45.4 | **49.5** | 32.7 |
| 3 | 44.9 | **47.8** | 42.1 |
| 5 | 36.9 | 39.2 | **42.3** |
| 7 | 33.1 | 35.3 | **40.0** |

`batched` wins at the default K=3 (verify processes N=K+1 tokens); `dense` is
weight-load-bound (~flat in N) and overtakes at K≥5. See `_moe_bench.py` for the
isolated MoE microbenchmark and `_sweep.py` for the end-to-end sweep. (This table
uses a harder prompt than the K-sweep above, so absolute tok/s is lower; the
per-K kernel ranking is what's robust.)

## How it works

### Speculation loop

```
1. Target prefill on prompt → first token + 3 tapped hidden states
2. Drafter prefill on prompt (EAGLE shift) → drafter KV cache over positions 0..P-2
3. Loop:
   a. Drafter: roll cache back to last accepted pos; run
      [newly-accepted tokens | K-1 ptd slots] in ONE parallel forward pass,
      attending to the FULL drafter KV cache → K draft tokens
   b. Target verify: run [last_accepted, draft_0, ..., draft_{K-1}] through
      target layers (seq_len = K+1) with block-causal mask
   c. Accept: longest prefix where draft[i] == target_argmax[i]
   d. Emit accepted tokens + bonus correction token
   e. Advance KV cache position by (accepted + 1)
```

### P-EAGLE parallel drafting (K tokens in one pass)

Unlike autoregressive EAGLE which runs K sequential drafter passes, P-EAGLE
generates all K draft tokens simultaneously. Per step, the slots appended to the
drafter's KV cache (all attending to the full prior context) are:

| Slot | Embedding input | Hidden state input |
|------|----------------|-------------------|
| committed (incl. NTP, depth 0) | `embed(newly-accepted token)` | `fc(concat(aux_layer_1, aux_layer_11, aux_layer_20))` — real target hidden |
| K-1 MTP (depth 1..K-1) | `embed(ptd_token_id)` — placeholder | `fc(mask_hidden)` — learnable shared hidden |

The K draft logits come from the **last committed slot** (NTP) plus the K-1 ptd
slots (MTP), each through the EAGLE-3 fusion midlayer + 3 plain Llama decoder
layers. All positions attend causally (over absolute positions) to the full
context KV cache — not just to each other.

### Architecture details

The P-EAGLE drafter (`GPT-OSS-20B-P-EAGLE`, ~3.6 GB bf16):

| Component | Description |
|-----------|-------------|
| `fc` (8640→2880) | Fuses 3 target hidden states (layers 2, 12, 21 of 24-layer target) |
| `midlayer` | EAGLE-3 fusion decoder layer: attention takes 2×hidden (embed⊕hidden), has `hidden_norm` |
| `layers.1/2/3` | Plain Llama decoder layers (SiLU MLP, llama3 RoPE) |
| `mask_hidden` (1,1,8640) | Learnable shared hidden state for MTP positions |
| `ptd_token_id` = 201020 | Placeholder token whose embedding fills MTP positions |
| `d2t` / `t2d` | Draft↔target vocab mapping (identity for this checkpoint) |
| `lm_head` (2880→201088) | Full target vocab, replicated on every rank |

### Verification

The target verifies K+1 candidate tokens in a single multi-token forward pass:
- Runs the full gpt-oss decoder stack with `seq_len = K+1` at a runtime offset
- Uses absolute-position RoPE and a block-causal attention mask
- Writes K+1 new KV cache entries contiguously
- Produces per-position greedy argmax via cross-rank reduction

**Greedy acceptance makes KV rollback implicit**: rejected speculative entries are
overwritten by the next verify pass, and the causal mask prevents any query from
attending past its own position.

## Files

| File | Purpose |
|------|---------|
| `speculate.py` | Main entry: speculation loop orchestrating target + drafter |
| `config.py` | `EagleConfig` for the P-EAGLE drafter (llama3 RoPE, fc, mask_hidden, K) |
| `tensor_preparation.py` | Convert P-EAGLE checkpoint to x@W form (replicated, no TP) |
| `drafter_model.py` | Device-side drafter: loads weights, compiles kernel, runs draft |
| `kernels/drafter.py` | Parallel-drafting forward kernel (K tokens in one pass) |
| `kernels/drafter_layer.py` | EAGLE-3 fusion midlayer + plain Llama layers |
| `kernels/verify.py` | Multi-position greedy argmax for verification |
| `kernels/rope.py` | llama3 RoPE (different from target's YaRN RoPE) |
| `kernels/rmsnorm.py`, `softmax.py` | Leaf kernels (copied from base) |

## Validation

| What | Result |
|------|--------|
| Drafter layer math (`DrafterCPU`) | ✅ cos 0.9999 vs vLLM on prefill (85/86 tokens) |
| KV-cached decode steps | ✅ cos 0.9999, 100% draft-token match vs vLLM |
| `draft()` public API | ✅ 7/7 draft tokens match vLLM on multiple decode steps |
| Aux tap layers (1, 11, 20) | ✅ fc chunks equal target layer outputs at cos 1.0 |
| Rollback / context-attention | ✅ guarded by `test_drafter_cpu.py` |

## Acceptance length: root cause & fix

Earlier acceptance was ~1.4 tokens/step (vs the model card's 3.30–3.80 at K=7).
This was root-caused on GPU by running the **identical checkpoint** through vLLM's
`eagle3` parallel-drafting path and capturing its exact drafter I/O, then
reproducing it with a standalone PyTorch reference (cosine **0.9999**, 100%
draft-token match). Three issues were found and fixed:

1. **Context-blind drafting (dominant).** `speculate.py` drove `DrafterModel`
   (`kernels/drafter.py`), which runs only the K draft positions under a (K,K)
   cross-depth mask with **no prefill and no KV cache** — the MTP slots never saw
   the prompt, so they produced generic tokens. The drafter must keep a KV cache
   over the whole context and have every new position attend to it (plain causal
   over absolute positions). `speculate.py` now uses the KV-cached `DrafterCPU`:
   it prefills the drafter on the prompt and, each step, rolls the cache back to
   the last accepted position and runs `[newly-accepted tokens | K-1 ptd slots]`.

2. **`rollback()` truncated the wrong axis.** The cache tensors are
   `(B, n_kv, seq, head_dim)`; `rollback()` sliced dim 1 (`n_kv`) instead of dim 2
   (`seq`), so rejected speculative KV entries were never discarded and corrupted
   every later step. Fixed to slice the sequence axis. Guarded by
   `test_drafter_cpu.py::test_rollback_restores_clean_cache`.

3. **Aux tap off-by-one.** vLLM's EAGLE-3 default `(2, n//2, n-3)` captures the
   residual stream *entering* those layers (`layer_idx=i+1` after layer `i`), i.e.
   the **outputs of layers (1, 11, 20)** for the 24-layer target. Our prefill loop
   captures *after* layer `i`, so `default_aux_layers` now returns `(1, 11, 20)`.
   Verified on GPU: the drafter's 3 fc chunks equal target layer outputs (1,11,20)
   at cosine 1.0.

**Prompt formatting matters too.** The drafter is trained on chat data; raw
completion prompts are out-of-distribution and roughly halve acceptance (GPU,
identical checkpoint, K=7: **3.65** chat vs **1.99** raw). `speculate.py` now
applies the chat template by default (`--raw-prompt` to opt out).

### Validated algorithm (matches vLLM)

1. Target taps the residual stream (`x+residual`) at the **outputs of layers
   (1, 11, 20)**; concat the 3 → `fc` → H.
2. **EAGLE shift:** drafter slot `p` pairs `embed(token@p+1)` with
   `target_hidden@p`.
3. **Drafter prefill** over the prompt builds a KV cache (positions 0..P-2), plain
   causal.
4. **Each draft step:** roll the cache back to the last accepted position; append
   `[newly-accepted tokens (real target hidden) | K-1 ptd slots (fc(mask_hidden),
   ptd_token_id embedding)]` at consecutive absolute positions, attending to the
   full cache; the K draft logits are the last committed slot (NTP) + the K-1 ptd
   slots (MTP).

### vLLM reference (parallel_drafting)

vLLM produces all K tokens in **one forward pass**: the expanded input is
`[shifted context tokens | bonus (next_token) | K-1 ptd_token positions]`, all run
together. `parallel_drafting_hidden_state_tensor = fc(mask_hidden)` fills the MTP
positions; the `copy_and_expand_eagle_inputs_kernel` lays out sequential positions
(`start_pos + j`) and tags parallel-draft slots with `ptd_token_id`. Only the
midlayer concatenates embeds with hidden to 2H; later layers are standard.

### Status / remaining work

The CPU drafter path (`DrafterCPU`) and the `speculate.py` loop bookkeeping are
GPU-validated against vLLM. The **on-device drafter keeps a KV cache and runs as
a single fused kernel** (`drafter_model.py` + `drafter_fused_kernel` in
`kernels/drafter.py`): fc-fusion + (embed, hidden) assembly + all 4 layers +
final norm + lm_head in one device call, each layer aliasing its own cache. It
prefills over the prompt and drafts K context-attending positions per step,
matching `DrafterCPU` token-for-token on device (parity is sub-noise bf16 ties,
logit cosine 0.9999) and reaching acceptance ~4.3 at ~50 decode tok/s on trn2
(TP=4). It is the default; `--cpu-drafter` selects the reference path.

Each draft step runs a **single fused forward** of width `W = C + K - 1` over
`[commit_0 .. commit_{C-1} | ptd_0 .. ptd_{K-2}]` (accepted tokens + MTP slots), so
committing accepted tokens and drafting the next K happen in one kernel — no
separate per-token commit or per-layer launches. One kernel is compiled per width
(`W` ranges `K..2K`) at load time.

Remaining work (profile with `SPEC_PROFILE=1`):

- **Verify is the dominant per-step cost** (~70–78% depending on K). Profiling
  shows it scales as **~19 ms fixed + ~6.3 ms/candidate token**, and it is genuine
  device layer compute (argmax/head and aux copies are <3 ms/step combined), not
  host overhead. The fixed ~19 ms is a single-token forward — dominated by
  streaming the 24 layers' weights from HBM — so it is amortized across the batch.
  Tuning **K to the acceptance plateau** (K=3 here) is the highest-leverage lever;
  a stronger drafter (higher plateau) would let larger K pay off. Further gains
  would come from the shared target MoE `feedforward` kernel (e.g. gathering each
  routed expert's weights once per step instead of per token).
- **Time-to-first-token.** `prefill()` compiles a full-prompt-width fused kernel
  on the first call (inside the timed region), which dominates TTFT on a cold
  build cache; a warm cache amortizes it. Pre-compiling a bucketed set of prompt
  widths (like the base model's context buckets) would hide it.
