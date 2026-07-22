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
Mean acceptance length: X.XX (K=3)
Decode tokens/sec: XX.XX
```

### Drafter (on device)

The drafter forward runs **on the Neuron device** (`kernels/drafter.py` +
`drafter_model.py`). It keeps a full per-layer KV cache on device — prefill over
the prompt, then each step commits the accepted tokens and drafts K positions
attending to the whole context — as a single fused kernel with no per-step
host↔device sync (~4.1 acceptance, ~42 decode tok/s on trn2 at TP=4, K=7, chat
prompt). Set `SPEC_PROFILE=1` to print a per-step draft/verify time breakdown.
Note: the first `prefill()` compiles a full-prompt-width fused kernel; `main()`
runs one warm-up prefill before timing so time-to-first-token (~0.5s) reflects
steady-state prefill, not that one-time build cost.

### Choosing K (draft tokens per step)

Verify runs `K+1` candidate tokens through the target in one pass. Its cost is
dominated by loading the 24 layers' weights from HBM (a fixed per-step cost) plus
a smaller per-candidate-token term, so verify is efficient *per token* and the
fixed part is amortized across all `K+1` tokens. Acceptance keeps rising with K
but with diminishing returns, while per-step verify cost grows, so throughput
peaks in the middle. Sweep (n=200, chat prompt, `reference` MoE, trn2 TP=4):

| K | Mean acceptance | Decode tok/s |
|---|-----------------|--------------|
| 1 | 1.90 | 40.7 |
| 2 | 2.68 | 46.7 |
| **3 (default)** | **3.17** | **48.2** |
| 4 | 3.62 | 48.5 |
| 7 | 4.08 | 42.5 |

**K=3–4 is the throughput optimum (~48 tok/s), beating both the base model
(~31 tok/s) and larger K.** K=4 edges K=3 by a hair here; K=3 is kept as the
default for its lower verify latency. Beyond K≈4, added verify time outpaces the
extra acceptance. Set `-k` to override.

### Speedup over the base model

Speedup = speculative tok/s ÷ base (non-speculative) tok/s of the same target,
**with the same MoE kernel on both sides** (the base model also honors
`GPT_OSS_MOE_KERNEL`, so comparing spec-batched against base-reference would
overstate the win). Measured on trn2 (TP=4, K=3, binary-search chat prompt,
acceptance ~3.2, n=200 each, both warm):

| MoE kernel | Base tok/s | P-EAGLE K=3 tok/s | Speedup |
|---|---|---|---|
| `reference` (default) | 31.5 | 48.2 | **1.53×** |
| `batched` | 34.7 | 50.8 | **1.47×** |

The speedup is below the ~3.2 acceptance length because verify is not free. At
K=3 (`SPEC_PROFILE=1`), each step spends ~52 ms in verify (running `K+1`=4 tokens
through the full 24-layer target, dominated by weight loads) and ~15 ms drafting.
Verifying 4 tokens costs well under 4× a single-token decode while producing ~3.2
tokens of output — hence a solid but sub-acceptance speedup. The `batched` MoE
lifts base and spec decode alike (31.5→34.7 base), so it barely moves the *ratio*
(1.53×→1.47×) even as it raises absolute tok/s.

**Speedup scales ~linearly with acceptance length, which is prompt-dependent.**
Chat-formatted, in-distribution prompts accept ~3.2 (→ ~1.5×); the same prompt
tokenized raw (`--raw-prompt`, out of the drafter's training distribution)
accepts only ~2.2 (→ ~1.05×, i.e. essentially no win). Always benchmark tok/s
with a fixed, representative prompt. The biggest lever on speedup is a stronger
drafter (raises acceptance); a cheaper verify (LNC2, or the `batched`/`dense`
MoE kernels below) is the secondary lever.

### MoE kernel selection

The target's MoE feed-forward has several implementations, selected via the
`GPT_OSS_MOE_KERNEL` env var (read in `kernels/transformer_layer.py`):

| value | description | best regime |
|---|---|---|
| `reference` (default) | per-(token, expert) Python loop | clearest baseline |
| `batched` | gather top-k experts, batched GEMV | decode / verify with K ≤ 4 |
| `dense` | all experts as one dense GEMM, router-masked | verify with K ≥ 5 |

All are numerically equivalent. End-to-end P-EAGLE tok/s (TP=4, n=200,
binary-search chat prompt):

| K | `reference` | `batched` | `dense` |
|---|-----------|-----------|---------|
| 1 | 40.7 | **41.4** | 29.7 |
| 3 | 48.2 | **50.8** | 45.3 |
| 5 | 47.9 | 52.0 | **53.3** |
| 7 | 42.5 | 44.1 | **48.6** |

`batched` wins at the default K=3 (verify processes N=K+1 tokens); `dense` is
weight-load-bound (~flat in N) and overtakes at K≥5, where it also lifts the
optimum to ~53 tok/s. The per-K kernel ranking is the robust takeaway.

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
| `fc` (8640→2880) | Fuses 3 target hidden states (outputs of layers 1, 11, 20 of 24-layer target) |
| `midlayer` | EAGLE-3 fusion decoder layer: attention takes 2×hidden (embed⊕hidden), has `hidden_norm` |
| `layers.1/2/3` | Plain Llama decoder layers (SiLU MLP, llama3 RoPE) |
| `mask_hidden` (1,1,8640) | Learnable shared hidden state for MTP positions |
| `ptd_token_id` (≈201020, from checkpoint) | Placeholder token whose embedding fills MTP positions |
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
