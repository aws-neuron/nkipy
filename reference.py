import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from config import Config
from convert_torch_numpy_dtype import numpy_to_torch_type

# from .weights import Checkpoint


class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )
        torch.nn.init.normal_(self.scale, mean=1.0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        initial_context_length: int = Config.initial_context_length,
        scaling_factor: float = Config.rope_scaling_factor,
        ntk_alpha: float = Config.rope_ntk_alpha,
        ntk_beta: float = Config.rope_ntk_beta,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: Config,
        sliding_window: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.n_heads
        self.num_key_value_heads = config.n_kv_heads
        # Only apply sliding window to every other layer
        self.sliding_window = sliding_window
        self.sinks = torch.nn.Parameter(
            torch.randn(config.n_heads, device=device, dtype=numpy_to_torch_type[config.dtype])
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.n_heads + 2 * config.n_kv_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=numpy_to_torch_type[config.dtype]
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.n_heads,
            config.hidden_size,
            device=device,
            dtype=numpy_to_torch_type[config.dtype],
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            base=config.rope_theta,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads * self.head_dim : (
                self.num_attention_heads + self.num_key_value_heads
            )
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (
                self.num_attention_heads + 2 * self.num_key_value_heads
            )
            * self.head_dim,
        ].contiguous()

        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        t = self.out(t)
        t = x + t
        return t, k, v


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    # to match neuron implementation
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        config: Config,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=numpy_to_torch_type[config.dtype]
        )
        assert config.intermediate_size % self.world_size == 0
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=numpy_to_torch_type[config.dtype],
            )
        )
        torch.nn.init.xavier_uniform_(self.mlp1_weight)
        self.mlp1_bias = torch.nn.Parameter(
            torch.randn(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=numpy_to_torch_type[config.dtype],
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=device,
                dtype=numpy_to_torch_type[config.dtype],
            )
        )
        torch.nn.init.xavier_uniform_(self.mlp2_weight)
        self.mlp2_bias = torch.nn.Parameter(
            torch.randn(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=numpy_to_torch_type[config.dtype],
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        # MLP #1
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t += mlp2_bias

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        return x + t


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: Config,
        sliding_window: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.attn = AttentionBlock(config, sliding_window, device)
        self.mlp = MLPBlock(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, k, v = self.attn(x)
        x = self.mlp(attn_out)
        return x, k, v, attn_out


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: Config,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=numpy_to_torch_type[config.dtype]
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, config.sliding_window if layer_idx % 2 == 0 else 0, device)
                for layer_idx in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=numpy_to_torch_type[config.dtype],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = Config(**json_config)

        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()

        # Load weights
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_intermediate_size = config.intermediate_size // world_size

        checkpoint = Checkpoint(path, device)

        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)

            # Note: it would be more efficient to do sharding before upcasting from MXFP4,
            # but for simplicity we do it after.
            if "mlp1" in name:  # both weight and bias
                loaded_tensor = loaded_tensor[
                    :,
                    my_rank * 2 * per_rank_intermediate_size : (my_rank + 1)
                    * 2
                    * per_rank_intermediate_size,
                    ...,
                ]
            elif "mlp2_weight" in name:  # only weight
                loaded_tensor = loaded_tensor[
                    ...,
                    my_rank * per_rank_intermediate_size : (my_rank + 1)
                    * per_rank_intermediate_size,
                ]
            try:
                param.data.copy_(loaded_tensor)
            except:
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        return model
