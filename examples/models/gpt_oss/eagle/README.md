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
python eagle/tensor_preparation.py \
    --model-name /path/to/GPT-OSS-20B-P-EAGLE \
    --output-dir ./eagle/tmp_p-eagle
```

### 2. Run speculative decoding

``` sh
TP=4
torchrun --nproc-per-node $TP eagle/speculate.py \
    --target-checkpoint ./tmp_gpt-oss-20b \
    --draft-checkpoint ./eagle/tmp_p-eagle \
    --model /path/to/gpt-oss-20b \
    --draft-model /path/to/GPT-OSS-20B-P-EAGLE \
    -n 256 -k 7 \
    "Write a Python function that implements binary search."
```

Output includes acceptance metrics:

```
Time to first token: 0.6s
Generated 256 tokens in N verify steps
Mean acceptance length: X.XX (K=7)
Decode tokens/sec: XX.XX
```

## How it works

### Speculation loop

```
1. Target prefill on prompt → first token + 3 tapped hidden states
2. Seed: run first token through target decode → hidden states for drafter
3. Loop:
   a. Drafter: K tokens in ONE parallel forward pass
   b. Target verify: run [last_accepted, draft_0, ..., draft_{K-1}] through
      target layers (seq_len = K+1) with block-causal mask
   c. Accept: longest prefix where draft[i] == target_argmax[i]
   d. Emit accepted tokens + bonus correction token
   e. Advance KV cache position by (accepted + 1)
```

### P-EAGLE parallel drafting (K tokens in one pass)

Unlike autoregressive EAGLE which runs K sequential drafter passes, P-EAGLE
generates all K draft tokens simultaneously:

| Position | Embedding input | Hidden state input |
|----------|----------------|-------------------|
| 0 (NTP) | `embed(last_accepted_token)` | `fc(concat(aux_layer_2, aux_layer_12, aux_layer_21))` — real target hidden |
| 1..K-1 (MTP) | `embed(ptd_token_id)` — placeholder | `fc(mask_hidden)` — learnable shared hidden |

All K positions attend under a **cross-depth causal mask** (depth d sees depths
≤ d) through the EAGLE-3 fusion midlayer + 3 plain Llama decoder layers. Each
position's `lm_head` logit gives one draft token.

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
| Drafter kernel math | ✅ All 7 draft tokens match independent PyTorch reference |
| Speculation output | ✅ Lossless — output matches HF greedy baseline exactly |
| Drafter with HF hidden states | ✅ Draft[0] matches target greedy perfectly |
| Multi-token verify | ✅ Block-causal mask + KV scatter correct |

## Known limitation: acceptance length

The current CPU-side acceptance length is ~1.4 tokens/step (vs paper's ~3.7 for
GPT-OSS 20B at K=5). The NTP (depth 0) position works correctly — draft[0]
frequently matches the target's greedy. The MTP (depth 1+) positions
underperform, producing generic tokens instead of context-specific ones.

### What's been verified

- Drafter NTP produces the correct next token when given HF hidden states ✅
- Drafter KV cache is necessary and improves acceptance from 1.0 to 1.4 ✅
- The EAGLE shifted-token convention (input_ids shifted +1 vs hidden states)
  matches vLLM's implementation ✅
- Hidden-state capture point (output of tap layers) matches what vLLM uses ✅
- The midlayer concat `[norm(embed), norm(hidden)]` → 2H → attention → H output
  with H-wide residual matches vLLM's `Eagle3DecoderLayer` ✅

### vLLM reference (parallel_drafting)

Studied from the installed vLLM at `private-vllm-neuron/.venv`. Key findings:

1. vLLM's parallel drafting produces ALL K tokens in **one forward pass**: the
   expanded input contains [shifted context tokens | bonus (next_token) | K-1
   ptd_token positions]. All go through the model together with PagedAttention.

2. The `parallel_drafting_hidden_state_tensor` = `fc(mask_hidden)` (the fc-fused
   mask_hidden at 2880 dim), placed at the MTP positions in the hidden_states
   input to the model.

3. The Triton kernel `copy_and_expand_eagle_inputs_kernel` handles the layout:
   positions are sequential (start_pos + j), parallel-draft slots get
   `ptd_token_id` for input_ids and `parallel_drafting_hidden_state_tensor` for
   hidden_states.

4. The model's `forward(input_ids, positions, hidden_states)` takes all three
   as separate tensors of the same length. Only the midlayer (layer 0)
   concatenates embeds with hidden to produce 2H; subsequent layers are standard.

### Remaining gap to investigate

The MTP positions have correct architecture but produce generic predictions. The
most likely remaining issue is an off-by-one in how the drafter's RoPE positions
map to the target's absolute positions during the KV-cached speculation loop.
vLLM assigns positions sequentially from the start of the context, and the
parallel-draft positions get positions immediately following the last valid token.
Our `drafter_cpu.py` does the same via `torch.arange(cache_len, cache_len + K)`.

### Path forward

1. Run the drafter via vLLM on GPU with this exact checkpoint and capture the
   actual acceptance length (confirms the checkpoint quality ceiling)
2. If vLLM achieves ~3.7, the issue is in our inference loop (position/hidden
   alignment during rollback+extend)
3. If vLLM also gets ~1.4, the checkpoint may be under-trained for this prompt
   distribution (the paper evaluates on HumanEval/MT-Bench/GSM-8K specifically)
