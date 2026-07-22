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

Run (from the gpt_oss/ directory, with peagle/ on PYTHONPATH):
    torchrun --nproc-per-node $TP peagle/speculate.py \
        --target-checkpoint ./tmp_gpt-oss-20b \
        --draft-checkpoint ./tmp_p-eagle \
        --model openai/gpt-oss-20b \
        --draft-model amazon/GPT-OSS-20B-P-EAGLE \
        -n 200 -k 7 "The capital of France is"
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

# Both the base gpt_oss package and the peagle subpackage have flat `config.py` /
# `kernels/` modules. We put the base dir (gpt_oss/) FIRST on sys.path so the base
# flat modules win, and import everything peagle-specific as the `peagle.*` package
# (which can't collide). Works when run as `peagle/speculate.py` from gpt_oss/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
# torchrun puts the script's own dir (peagle/) on sys.path[0], which would shadow
# the base flat modules (config.py, kernels/) with peagle's. Drop it and put the
# base dir first; peagle code is reached via the `peagle.*` package from _BASE.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from config import Config, get_config  # noqa: E402  (base config; _BASE wins)
from kernels.transformer_layer import transformer_layer  # noqa: E402  (base)
from nkipy.runtime import DeviceKernel, DeviceTensor  # noqa: E402
from peagle.config import get_eagle_config  # noqa: E402
from peagle.drafter_model import DrafterModel  # noqa: E402
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
        from peagle.kernels.verify import verify_argmax

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

        _prof = os.environ.get("SPEC_PROFILE") == "1"
        if _prof:
            _t = time.time()
        aux = []
        for i in range(cfg.num_layers):
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
            if cfg.aux_layers is not None and i in cfg.aux_layers:
                if _prof:
                    _ta = time.time()
                aux.append(h.torch().clone())
                if _prof:
                    self._t_aux = getattr(self, "_t_aux", 0.0) + time.time() - _ta
        if _prof:
            self._t_layers = getattr(self, "_t_layers", 0.0) + (time.time() - _t)
            _t = time.time()

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
        out = target_ids.torch().reshape(S).tolist()
        if _prof:
            self._t_argmax = getattr(self, "_t_argmax", 0.0) + time.time() - _t
        return out, aux


def _stack_aux(aux_list):
    """Concatenate the 3 tap-layer hiddens along the feature axis: (B, S, 3H)."""
    return torch.cat([a.to(torch.bfloat16) for a in aux_list], dim=-1)


class _DeviceDrafterAdapter:
    """Thin wrapper exposing the on-device `DrafterModel` under the loop's
    interface (`prefill` / `draft(pending_tokens, pending_aux3, base_pos)`).

    `DrafterModel` keeps a full per-layer KV cache on device, so the adapter just
    forwards the calls: it prefills the drafter over the prompt and, each step,
    commits the accepted tokens and drafts K positions attending to the full
    context.
    """

    def __init__(self, draft_model, draft_checkpoint, target_hidden_size, K):
        cfg = get_eagle_config(draft_model, target_hidden_size, num_draft_tokens=K)
        weights = load_file(
            os.path.join(draft_checkpoint, "drafter.safetensors"), device="cpu"
        )
        build_dir = os.path.abspath(os.path.join(draft_checkpoint, "build_device"))
        os.makedirs(build_dir, exist_ok=True)
        self._model = DrafterModel(weights, cfg, build_dir)

    def prefill(self, token_ids, aux_hidden_states):
        self._model.prefill(token_ids, aux_hidden_states)

    def draft(self, commit_token_ids, commit_aux3, base_pos):
        return self._model.draft(commit_token_ids, commit_aux3, base_pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max-new-tokens", type=int, default=128)
    # K=3 is the measured throughput optimum on trn2: verify cost grows ~6.3
    # ms/token but acceptance plateaus around 3.3 for this drafter, so larger K
    # pays more verify time for no extra accepted tokens. See README.
    parser.add_argument("-k", "--num-draft-tokens", type=int, default=3)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Tokenize the prompt as raw text instead of applying the chat "
        "template (chat formatting matches the drafter's training distribution "
        "and yields higher acceptance).",
    )
    parser.add_argument("--target-checkpoint", default="./tmp_gpt-oss-20b")
    parser.add_argument("--draft-checkpoint", default="./tmp_p-eagle")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--draft-model", default="amazon/GPT-OSS-20B-P-EAGLE"
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    # Overridable so concurrent runs on one host don't collide on the comm port
    # (export NEURON_RT_ROOT_COMM_ID=localhost:<other_port> for a second instance).
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "localhost:61239")
    dist.init_process_group()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())

    K = args.num_draft_tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The drafter is trained on chat-formatted data; raw completion prompts are
    # out-of-distribution and roughly halve acceptance length (≈2.0 vs ≈3.7 at
    # K=7 on GPU). Apply the chat template unless --raw-prompt is set.
    if args.raw_prompt or tokenizer.chat_template is None:
        input_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"]
    else:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            return_tensors="np",
            return_dict=True,
        )["input_ids"]
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

    # Drafter (replicated on every rank). The on-device drafter keeps its own KV
    # cache over the full context and proposes K tokens in one parallel forward
    # pass per step (validated against vLLM's eagle3 parallel-drafting path). It
    # is tiny (4 layers, ~3.6 GB).
    print_log("Loading drafter weights")
    drafter = _DeviceDrafterAdapter(
        args.draft_model, args.draft_checkpoint, config.hidden_size, K
    )

    # ── Prefill the target on the prompt ──
    dist.barrier()
    t0 = time.time()
    hidden, aux = target.run_prefill(input_ids, capture_aux=True)
    # aux[k]: (1, prompt_len, H) for the k-th tap layer. Stack -> (1, prompt_len, 3H).
    prompt_aux3 = _stack_aux(aux)

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

    # ── Prefill the DRAFTER over the prompt (EAGLE shift) ──
    # The drafter pairs embed(token_{p+1}) with target_hidden(token_p) at slot p.
    # Prompt tokens t_0..t_{P-1} with prefill hiddens h_0..h_{P-1} fill drafter
    # slots 0..P-2 as (embed(t_{p+1}), h_p). Tokens are the prompt shifted by +1.
    P = prompt_len
    shifted_tokens = torch.as_tensor(input_ids).reshape(-1)[1:P]  # t_1..t_{P-1}
    drafter.prefill(shifted_tokens, prompt_aux3[:, : P - 1, :])

    # The next committed slot is the first generated token `next_id` at abs pos
    # P-1, paired with the prefill hidden at the last prompt position (h_{P-1}).
    pending_tokens = [next_id]
    pending_aux3 = prompt_aux3[:, P - 1 : P, :]  # (1, 1, 3H)
    cur_pos = P  # absolute position of the token that `next_id` will predict

    ttft = time.time() - t0
    n_accepted_total = 0
    n_steps = 0

    if dist.get_rank() == 0:
        print(f"\n{args.prompt}", end="")
        print(tokenizer.decode([next_id]), end="")
        sys.stdout.flush()

    t_decode = time.time()
    prof = {"draft": 0.0, "verify": 0.0}
    profile = os.environ.get("SPEC_PROFILE") == "1"
    while len(generated) < args.max_new_tokens:
        # 1) Draft K tokens. Commit the pending (newly accepted) tokens into the
        #    drafter's KV cache; the NTP prediction comes from the last committed
        #    slot and K-1 MTP (ptd) slots follow. base_pos is the absolute position
        #    of the first pending token.
        base_pos = cur_pos - len(pending_tokens)
        if profile:
            dist.barrier()
            _t = time.time()
        drafts = drafter.draft(pending_tokens, pending_aux3, base_pos)
        if profile:
            dist.barrier()
            prof["draft"] += time.time() - _t
            _t = time.time()

        # 2) Verify: feed [last_token, drafts...] at absolute cur_pos.
        cand = [generated[-1]] + drafts  # length K+1
        target_ids, aux = target.verify(cand, cur_pos)
        if profile:
            dist.barrier()
            prof["verify"] += time.time() - _t

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
        n_emitted = 0
        for tok in accepted:
            if len(generated) >= args.max_new_tokens:
                stop = True
                break
            generated.append(tok)
            n_emitted += 1
            if dist.get_rank() == 0:
                print(tokenizer.decode([tok]), end="")
                sys.stdout.flush()
            if tok in eos_ids:
                stop = True
                break
        if stop:
            break

        # 4) Advance. The accepted tokens become the drafter's committed slots for
        #    the next step. The verify window ran candidates at absolute positions
        #    cur_pos .. cur_pos+K, so its captured hidden at window index i is the
        #    target hidden at position cur_pos+i. Accepted token j (= target_ids[j])
        #    is the token predicted by window hidden j, i.e. it sits at sequence
        #    position cur_pos+j+1, and the drafter pairs it (EAGLE shift) with the
        #    target hidden at position cur_pos+j == verify window index j.
        #    Verified against vLLM: commit token j pairs with verify_aux3[:, j].
        verify_aux3 = _stack_aux(aux)  # (1, K+1, 3H)
        pending_tokens = list(accepted[:n_emitted])
        pending_aux3 = verify_aux3[:, :n_emitted, :]
        cur_pos += n_emitted

    decode_time = time.time() - t_decode
    if dist.get_rank() == 0:
        n_new = len(generated)
        accept_len = n_accepted_total / max(n_steps, 1)
        print(f"\n\nTime to first token: {ttft:.2f}s")
        print(f"Generated {n_new} tokens in {n_steps} verify steps")
        print(f"Mean acceptance length: {accept_len:.2f} (K={K})")
        print(f"Decode tokens/sec: {n_new / max(decode_time, 1e-6):.2f}")
        if profile:
            other = decode_time - prof["draft"] - prof["verify"]
            print(
                f"\n[profile] per-step avg over {n_steps} steps:"
                f"\n  draft : {1000 * prof['draft'] / n_steps:7.1f} ms/step "
                f"({100 * prof['draft'] / decode_time:.0f}%)"
                f"\n  verify: {1000 * prof['verify'] / n_steps:7.1f} ms/step "
                f"({100 * prof['verify'] / decode_time:.0f}%)"
                f"\n  other : {1000 * other / n_steps:7.1f} ms/step "
                f"({100 * other / decode_time:.0f}%)"
            )
            m = getattr(drafter, "_model", None)
            if m is not None and hasattr(m, "_t_forward"):
                print(
                    f"  drafter forward: {1000 * m._t_forward / n_steps:6.1f} ms/step "
                    f"(1 fused kernel/step)"
                )
            if hasattr(target, "_t_layers"):
                print(
                    f"  verify breakdown:"
                    f"\n    layers: {1000 * target._t_layers / n_steps:6.1f} ms/step "
                    f"(incl. aux copies {1000 * target._t_aux / n_steps:.1f})"
                    f"\n    argmax: {1000 * target._t_argmax / n_steps:6.1f} ms/step "
                    f"(head + argmax + host transfer)"
                )


if __name__ == "__main__":
    main()
