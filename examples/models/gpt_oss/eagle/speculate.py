"""P-EAGLE speculative decoding for gpt-oss.

Orchestrates target + drafter:

  1. Target prefill on the prompt, capturing the 3 EAGLE-3 tap-layer hidden
     states and the first real next token.
  2. Drafter proposes K tokens in one parallel forward pass from those hidden
     states + the last accepted token.
  3. Target verifies the K candidates in ONE multi-token forward pass (seq_len =
     K+1: the last accepted token followed by the K drafts), capturing fresh
     tap-layer hidden states and the target's greedy token at each position.
  4. Accept the longest prefix of drafts that matches the target's greedy tokens;
     append the target's correction token. Advance the KV write position by the
     number of accepted tokens + 1.

Greedy verification makes KV rollback implicit: rejected speculative KV entries
are simply overwritten by the next verify pass (which re-reads from the accepted
position), and the causal mask never lets a query attend past its own position.

Run (from the gpt_oss/ directory, with eagle/ on PYTHONPATH):
    torchrun --nproc-per-node $TP eagle/speculate.py \
        --target-checkpoint ./tmp_gpt-oss-20b \
        --draft-checkpoint ./tmp_p-eagle \
        --model /home/ubuntu/models/gpt-oss-20b \
        --draft-model /home/ubuntu/models/GPT-OSS-20B-P-EAGLE \
        -n 200 -k 7 "The capital of France is"
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

# Both the base gpt_oss package and the eagle subpackage have flat `config.py` /
# `kernels/` modules. We put the base dir (gpt_oss/) FIRST on sys.path so the base
# flat modules win, and import everything eagle-specific as the `eagle.*` package
# (which can't collide). Works when run as `eagle/speculate.py` from gpt_oss/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
# torchrun puts the script's own dir (eagle/) on sys.path[0], which would shadow
# the base flat modules (config.py, kernels/) with eagle's. Drop it and put the
# base dir first; eagle code is reached via the `eagle.*` package from _BASE.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from config import Config, get_config  # noqa: E402  (base config; _BASE wins)
from eagle.config import get_eagle_config  # noqa: E402
from eagle.drafter_model import DrafterModel  # noqa: E402
from kernels.transformer_layer import transformer_layer  # noqa: E402  (base)
from nkipy.runtime import DeviceKernel, DeviceTensor  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from utils import print_log  # noqa: E402

import gpt_oss as base  # noqa: E402  (base model)


def _resolve_eos_ids(model_name, tokenizer):
    try:
        from transformers import GenerationConfig

        eos = GenerationConfig.from_pretrained(model_name).eos_token_id
    except Exception:
        eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


class SpeculativeGptOss(base.GptOssModel):
    """Target model + verify kernels for speculative decoding."""

    def __init__(self, weights, config, num_draft_tokens):
        self.num_draft_tokens = num_draft_tokens
        super().__init__(weights, config)
        self._prepare_verify_kernels()

    def _prepare_verify_kernels(self):
        """Compile a seq_len=(K+1) layer kernel per attention type, and a
        per-position verify-argmax kernel."""
        from eagle.kernels.verify import verify_argmax

        S = self.num_draft_tokens + 1
        cfg = self.config
        x_verify = DeviceTensor.from_numpy(
            np.empty((cfg.max_batch_size, S, cfg.hidden_size), dtype=cfg.dtype),
            "x_verify",
        )
        start_pos = DeviceTensor.from_numpy(np.empty((1), dtype=np.int32), "vs_pos")

        self.kernel_verify_layer = {}
        for sliding in (False, True):
            sw = cfg.sliding_window if sliding else None
            lt = self.layer_tensors[0]
            self.kernel_verify_layer[sliding] = DeviceKernel.compile_and_load(
                transformer_layer,
                name=f"verify_layer_{'sw' if sliding else 'full'}",
                x=x_verify,
                start_pos=start_pos,
                qkv_weight=lt["qkv_weight"],
                qkv_bias=lt["qkv_bias"],
                o_weight=lt["o_weight"],
                o_bias=lt["o_bias"],
                sinks=lt["sinks"],
                input_weight=lt["input_weight"],
                post_attention_weight=lt["post_attention_weight"],
                router_weight=lt["router_weight"],
                router_bias=lt["router_bias"],
                gate_up_weight=lt["gate_up_weight"],
                gate_up_bias=lt["gate_up_bias"],
                down_weight=lt["down_weight"],
                down_bias=lt["down_bias"],
                cache_k=lt["cache_k"],
                cache_v=lt["cache_v"],
                configs=cfg,
                sliding_window=sw,
                build_dir=base.BUILD_DIR,
                additional_compiler_args=cfg.additional_compiler_args_nkipy,
            )

        self.kernel_verify_argmax = DeviceKernel.compile_and_load(
            verify_argmax,
            name="verify_argmax",
            h=x_verify,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            configs=cfg,
            build_dir=base.BUILD_DIR,
            additional_compiler_args=cfg.additional_compiler_args_nkipy,
        )

    def verify(self, tokens, start_pos):
        """Run K+1 candidate tokens through the target at absolute `start_pos`.

        Args:
            tokens: list of K+1 token ids (last accepted token + K drafts).
            start_pos: absolute position of tokens[0] in the sequence.
        Returns:
            (target_tokens, aux) where target_tokens is a length-(K+1) list of the
            target's greedy next token at each position, and aux is the list of 3
            captured tap-layer hidden states (each (B, K+1, H), host tensors).
        """
        cfg = self.config
        S = len(tokens)
        h = DeviceTensor.from_torch(
            self.tok_embedding[torch.tensor(tokens)].reshape(1, S, cfg.hidden_size),
            "verify_h",
        )
        pos = DeviceTensor.from_numpy(
            np.array([start_pos], dtype=np.int32), "verify_pos"
        )

        aux = []
        for i in range(cfg.num_layers):
            # Tap the input residual of each aux layer (output of layer i-1),
            # matching run_prefill / vLLM's EAGLE-3 capture point.
            if cfg.aux_layers is not None and i in cfg.aux_layers:
                aux.append(h.torch().clone())
            lt = self.layer_tensors[i]
            kernel = self.kernel_verify_layer[cfg.is_sliding(i)]
            inputs = {key: lt[key] for key in base._LAYER_WEIGHT_KEYS}
            inputs["x"] = h
            inputs["start_pos"] = pos
            inputs["cache_k.must_alias_input"] = lt["cache_k"]
            inputs["cache_v.must_alias_input"] = lt["cache_v"]
            kernel(
                inputs=inputs,
                outputs={
                    "output0": h,
                    "cache_k": lt["cache_k"],
                    "cache_v": lt["cache_v"],
                },
            )

        target_ids = DeviceTensor.from_numpy(
            np.empty((1, S), dtype=np.int32), "tgt_ids"
        )
        self.kernel_verify_argmax(
            inputs={
                "h": h,
                "norm_weight": self.norm_weight,
                "lm_head_weight": self.lm_head_weight,
            },
            outputs={"output0": target_ids},
        )
        return target_ids.torch().reshape(S).tolist(), aux


def _stack_aux(aux_list):
    """Concatenate the 3 tap-layer hiddens along the feature axis: (B, S, 3H)."""
    return torch.cat([a.to(torch.bfloat16) for a in aux_list], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max-new-tokens", type=int, default=128)
    parser.add_argument("-k", "--num-draft-tokens", type=int, default=7)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--target-checkpoint", default="./tmp_gpt-oss-20b")
    parser.add_argument("--draft-checkpoint", default="./tmp_p-eagle")
    parser.add_argument("--model", default="/home/ubuntu/models/gpt-oss-20b")
    parser.add_argument(
        "--draft-model", default="/home/ubuntu/models/GPT-OSS-20B-P-EAGLE"
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"
    dist.init_process_group()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())

    K = args.num_draft_tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"]
    prompt_len = input_ids.shape[1]
    eos_ids = _resolve_eos_ids(args.model, tokenizer)

    # Target config + aux taps.
    config = get_config(args.model, prompt_len, args.max_new_tokens)
    config.aux_layers = Config.default_aux_layers(config.num_layers)

    print_log("Loading target weights")
    tgt_shard = os.path.join(
        args.target_checkpoint, f"shard_{dist.get_rank()}.safetensors"
    )
    target = SpeculativeGptOss(load_file(tgt_shard, device="cpu"), config, K)

    # Drafter (replicated on every rank).
    print_log("Loading drafter weights")
    ecfg = get_eagle_config(
        args.draft_model,
        target_hidden_size=config.hidden_size,
        num_draft_tokens=K,
        max_seq_len=config.max_seq_len,
    )
    draft_weights = load_file(
        os.path.join(args.draft_checkpoint, "drafter.safetensors"), device="cpu"
    )
    drafter = DrafterModel(draft_weights, ecfg, base.BUILD_DIR)

    # ── Prefill the target on the prompt ──
    dist.barrier()
    t0 = time.time()
    hidden, aux = target.run_prefill(input_ids, capture_aux=True)

    # First token: greedy sample from the prefill hidden (reuse the base kernel).
    first_id_dev = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "first_id")
    target.kernel_cte_greedy_sampling(
        inputs={
            "h": hidden,
            "norm_weight": target.norm_weight,
            "lm_head_weight": target.lm_head_weight,
        },
        outputs={"output0": first_id_dev},
    )
    next_id = int(first_id_dev.torch().reshape(-1)[0])

    generated = [next_id]
    cur_pos = prompt_len  # absolute position of `next_id`

    # The prefill's aux gives hidden states at position (prompt_len - 1), but the
    # drafter needs the hidden state at position `cur_pos` (after `next_id` has been
    # processed through the target). Run a single-token decode on `next_id` to get
    # the hidden state at `cur_pos`, capturing aux along the way.
    seed_h = DeviceTensor.from_torch(
        target.tok_embedding[torch.tensor([[next_id]])], "seed_h"
    )
    seed_pos = DeviceTensor.from_numpy(np.array([cur_pos], dtype=np.int32), "seed_pos")
    seed_aux = []
    for i in range(config.num_layers):
        if config.aux_layers is not None and i in config.aux_layers:
            seed_aux.append(seed_h.torch().clone())
        target._run_layer("tkg", i, seed_h, seed_pos)
    last_aux3 = _stack_aux([a[:, 0:1, :] for a in seed_aux])
    cur_pos += 1

    ttft = time.time() - t0
    n_accepted_total = 0
    n_steps = 0

    if dist.get_rank() == 0:
        print(f"\n{args.prompt}", end="")
        print(tokenizer.decode([next_id]), end="")
        sys.stdout.flush()

    t_decode = time.time()
    while len(generated) < args.max_new_tokens:
        # 1) Draft K tokens from the last accepted token + its tapped hiddens.
        drafts = drafter.draft(last_aux3, generated[-1])

        # 2) Verify: feed [last_token, drafts...] at absolute cur_pos.
        cand = [generated[-1]] + drafts  # length K+1
        target_ids, aux = target.verify(cand, cur_pos)

        # 3) Accept the longest matching prefix (greedy).
        accepted = []
        for i in range(K):
            accepted.append(target_ids[i])  # target's correction/confirmation
            if drafts[i] != target_ids[i]:
                break
        else:
            # all K matched; the (K+1)-th target token is a free bonus token
            accepted.append(target_ids[K])

        n_accepted = len(accepted)
        n_accepted_total += n_accepted
        n_steps += 1

        # Emit accepted tokens (truncate at max + stop on EOS).
        stop = False
        for tok in accepted:
            if len(generated) >= args.max_new_tokens:
                stop = True
                break
            generated.append(tok)
            if dist.get_rank() == 0:
                print(tokenizer.decode([tok]), end="")
                sys.stdout.flush()
            if tok in eos_ids:
                stop = True
                break
        if stop:
            break

        # 4) Advance. The accepted tokens occupy cur_pos .. cur_pos+n_accepted-1
        # (already written to the KV cache by verify). The tapped hiddens for the
        # last accepted token seed the next draft.
        last_kept_index = n_accepted - 1  # index into the verify window
        last_aux3 = _stack_aux(
            [a[:, last_kept_index : last_kept_index + 1, :] for a in aux]
        )
        cur_pos += n_accepted

    decode_time = time.time() - t_decode
    if dist.get_rank() == 0:
        n_new = len(generated)
        accept_len = n_accepted_total / max(n_steps, 1)
        print(f"\n\nTime to first token: {ttft:.2f}s")
        print(f"Generated {n_new} tokens in {n_steps} verify steps")
        print(f"Mean acceptance length: {accept_len:.2f} (K={K})")
        print(f"Decode tokens/sec: {n_new / max(decode_time, 1e-6):.2f}")


if __name__ == "__main__":
    main()
