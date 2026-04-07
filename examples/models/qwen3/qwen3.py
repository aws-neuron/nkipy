import os

from ..common.kernels.sampling import greedy_sampling
from ..common.model import BaseModel, generate_and_print, load_model
from .kernels.transformer_layer import transformer_layer

EOS_TOKEN_IDS = {151643, 151645}


class Qwen3Model(BaseModel):
    LAYER_WEIGHT_KEYS = [
        ("qkv_weight", "qkv_weight"),
        ("o_weight", "o_weight"),
        ("gate_up_weight", "gate_up_weight"),
        ("down_weight", "down_weight"),
        ("router_weight", "router_weight"),
        ("q_norm_weight", "q_norm_weight"),
        ("k_norm_weight", "k_norm_weight"),
        ("input_weight", "input_weight"),
        ("post_attention_weight", "post_attention_weight"),
    ]

    transformer_layer = staticmethod(transformer_layer)
    greedy_sampling = staticmethod(greedy_sampling)

    def _kernel_layer_args(self):
        L0 = self.layer_tensors[0]
        return {
            "qkv_weight": L0["qkv_weight"],
            "o_weight": L0["o_weight"],
            "input_weight": L0["input_weight"],
            "q_norm_weight": L0["q_norm_weight"],
            "k_norm_weight": L0["k_norm_weight"],
            "cache_k": L0["cache_k"],
            "cache_v": L0["cache_v"],
            "post_attention_weight": L0["post_attention_weight"],
            "router_weight": L0["router_weight"],
            "gate_up_weight": L0["gate_up_weight"],
            "down_weight": L0["down_weight"],
        }

    def _kernel_input_keys(self):
        return [
            ("qkv_weight", "qkv_weight"),
            ("o_weight", "o_weight"),
            ("input_weight", "input_weight"),
            ("q_norm_weight", "q_norm_weight"),
            ("k_norm_weight", "k_norm_weight"),
            ("cache_k.must_alias_input", "cache_k"),
            ("cache_v.must_alias_input", "cache_v"),
            ("post_attention_weight", "post_attention_weight"),
            ("router_weight", "router_weight"),
            ("gate_up_weight", "gate_up_weight"),
            ("down_weight", "down_weight"),
        ]


if __name__ == "__main__":
    import argparse

    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max-new-tokens", type=int, default=16)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--checkpoint", default="/kaena/qwen3_shards_30B_A3B_TP8")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"
    dist.init_process_group()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())

    model, tokenizer, weights, config, input_ids = load_model(Qwen3Model, args)

    dist.barrier()
    generate_and_print(model, args.prompt, input_ids, tokenizer, EOS_TOKEN_IDS)
