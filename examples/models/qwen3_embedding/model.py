"""Qwen3-Embedding-*B model with shared kernel architecture"""

from typing import Optional

import numpy as np
from config import Qwen3Config
from embedding_utils import last_token_pool
from kernels.attention import qwen3_attention_kernel
from kernels.ffn import feedforward_kernel
from kernels.rmsnorm import rmsnorm
from kernels.rope import compute_qwen3_cos_sin
from kernels.token_embedding import token_embedding
from layer import Qwen3Layer
from logger import get_logger
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import DeviceKernel, DeviceTensor

logger = get_logger()

additional_compiler_args = (
    " --lnc 1 --model-type transformer --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
    " --enable-mixed-precision-accumulation"  # control sbuf buffer dtype for accumulation
)

BACKEND = "hlo"
if BACKEND == "hlo":
    OUTPUT_PREFIX = "output"
else:
    raise ValueError(f"Unknown backend: {BACKEND}")


class Qwen3EmbeddingModel:
    """
    Qwen3-Embedding model for Trainium.

    Key features:
    - Shared kernels: All layers use the same compiled kernels
    - Grouped Query Attention
    - Q/K normalization before RoPE
    - Last token pooling for embedding extraction
    """

    def __init__(self, weights: dict, config: Qwen3Config):
        self.config = config

        # Extract and prepare global weights
        self.tok_embedding_device = DeviceTensor.from_torch(
            weights.pop("tok_embedding"), "tok_embedding"
        )
        self.norm_weight_device = DeviceTensor.from_torch(
            weights.pop("norm_weight"), "norm_weight"
        )

        # Precompute RoPE cos/sin tables
        cos, sin = compute_qwen3_cos_sin(
            max_model_len=config.max_model_len,
            head_dim=config.head_dim,
            theta=config.rope_theta,
        )
        self.cos = DeviceTensor.from_numpy(cos, "cos")
        self.sin = DeviceTensor.from_numpy(sin, "sin")

        # STEP 1: Compile shared kernels (once for all layers)
        self._compile_shared_kernels()

        # STEP 2: Initialize all layers with shared kernels
        self.layers = self._create_layers(weights)

        logger.info(f"Initialized Qwen3 model with {len(self.layers)} layers")

    def _compile_shared_kernels(self):
        """
        Compile kernels once that will be shared across all layers.

        Kernels compiled:
        1. Token embedding kernel
        2. Attention kernel (includes input norm, QKV proj, Q/K norm, RoPE, attention, output proj)
        3. RMSNorm kernel (for post-attention normalization)
        4. FFN kernel (gate/up projection, SiLU, down projection)
        5. Final norm kernel
        """
        logger.info("Compiling shared kernels...")

        # Sample tensors for kernel compilation
        batch_seq_tokens = self.config.max_batch_size * self.config.max_model_len
        hidden_states = np.empty(
            (batch_seq_tokens, self.config.hidden_size), dtype=self.config.dtype
        )

        # 1. Token embedding kernel
        logger.info("Compiling token embedding kernel...")
        self.token_embedding_kernel = DeviceKernel.compile_and_load(
            NKIPyKernel.trace(token_embedding, backend=BACKEND),
            name="qwen3_token_embedding",
            tok_embedding=np.empty(
                (self.config.vocab_size, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            token_ids=np.zeros(
                (self.config.max_batch_size, self.config.max_model_len), dtype=np.uint32
            ),
            additional_compiler_args=additional_compiler_args,
        )

        # 2. Attention kernel (shared across all layers)
        logger.info("Compiling attention kernel...")
        qkv_size = (
            self.config.num_attention_heads + 2 * self.config.num_key_value_heads
        ) * self.config.head_dim
        self.shared_attention_kernel = DeviceKernel.compile_and_load(
            kernel=NKIPyKernel.trace(qwen3_attention_kernel, backend=BACKEND),
            hidden_states=hidden_states,
            input_layernorm_weight=np.empty(
                self.config.hidden_size, dtype=self.config.dtype
            ),
            qkv_weight=np.empty(
                (self.config.hidden_size, qkv_size), dtype=self.config.dtype
            ),
            o_weight=np.empty(
                (
                    self.config.num_attention_heads * self.config.head_dim,
                    self.config.hidden_size,
                ),
                dtype=self.config.dtype,
            ),
            q_norm_weight=np.empty(self.config.head_dim, dtype=self.config.dtype),
            k_norm_weight=np.empty(self.config.head_dim, dtype=self.config.dtype),
            cos=self.cos,
            sin=self.sin,
            config=self.config,
            compute_dtype=self.config.dtype,
            name="qwen3_attention_shared",
            additional_compiler_args=additional_compiler_args,
        )

        # 3. RMSNorm kernel (shared for post-attention normalization)
        logger.info("Compiling RMSNorm kernel...")
        self.shared_rmsnorm_kernel = DeviceKernel.compile_and_load(
            NKIPyKernel.trace(rmsnorm, backend=BACKEND),
            x=hidden_states,
            weight=np.empty(self.config.hidden_size, dtype=self.config.dtype),
            eps=self.config.rms_norm_eps,
            name="qwen3_rmsnorm_shared",
            additional_compiler_args=additional_compiler_args,
        )

        # 4. FFN kernel (shared across all layers)
        logger.info("Compiling FFN kernel...")
        self.shared_ffn_kernel = DeviceKernel.compile_and_load(
            kernel=NKIPyKernel.trace(feedforward_kernel, backend=BACKEND),
            x=hidden_states,
            gate_up_weight=np.empty(
                (self.config.hidden_size, 2 * self.config.intermediate_size),
                dtype=self.config.dtype,
            ),
            down_weight=np.empty(
                (self.config.intermediate_size, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            gate_up_bias=np.zeros(
                2 * self.config.intermediate_size, dtype=self.config.dtype
            ),
            down_bias=np.zeros(self.config.hidden_size, dtype=self.config.dtype),
            name="qwen3_ffn_shared",
            additional_compiler_args=additional_compiler_args,
        )

        # 5. Final layer norm kernel
        logger.info("Compiling final norm kernel...")
        self.final_norm_kernel = DeviceKernel.compile_and_load(
            NKIPyKernel.trace(rmsnorm, backend=BACKEND),
            x=hidden_states,
            weight=self.norm_weight_device,
            eps=self.config.rms_norm_eps,
            name="qwen3_final_norm",
            additional_compiler_args=additional_compiler_args,
        )

        logger.info("All shared kernels compiled successfully!")

    def _create_layers(self, weights: dict) -> list:
        """Create all transformer layers with shared kernels and layer-specific weights"""
        layers = []

        for layer_id in range(self.config.num_hidden_layers):
            layer_prefix = f"layers.{layer_id}"

            # Extract layer-specific weights
            layer_weights = {
                "qkv_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.qkv_weight"), f"qkv_weight_L{layer_id}"
                ),
                "q_norm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.q_norm_weight"),
                    f"q_norm_weight_L{layer_id}",
                ),
                "k_norm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.k_norm_weight"),
                    f"k_norm_weight_L{layer_id}",
                ),
                "o_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.o_weight"), f"o_weight_L{layer_id}"
                ),
                "input_layernorm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.input_layernorm_weight"),
                    f"input_layernorm_weight_L{layer_id}",
                ),
                "post_attention_layernorm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.post_attention_layernorm_weight"),
                    f"post_attention_layernorm_weight_L{layer_id}",
                ),
                "gate_up_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.gate_up_weight"),
                    f"gate_up_weight_L{layer_id}",
                ),
                "down_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.down_weight"),
                    f"down_weight_L{layer_id}",
                ),
            }

            # Create layer with shared kernels
            layer = Qwen3Layer(
                layer_id=layer_id,
                config=self.config,
                cos=self.cos,
                sin=self.sin,
                shared_attention_kernel=self.shared_attention_kernel,
                shared_rmsnorm_kernel=self.shared_rmsnorm_kernel,
                shared_ffn_kernel=self.shared_ffn_kernel,
                **layer_weights,
            )

            layers.append(layer)

        return layers

    def forward(
        self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)

        Returns:
            embeddings: Extracted embeddings [batch_size, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Truncate if sequence is too long
        if seq_len > self.config.max_model_len:
            input_ids = input_ids[:, : self.config.max_model_len]
            seq_len = self.config.max_model_len

        # Create or adjust attention mask
        if attention_mask is None:
            attention_mask = np.ones((batch_size, seq_len), dtype=np.float32)
        elif attention_mask.shape[1] > self.config.max_model_len:
            attention_mask = attention_mask[:, : self.config.max_model_len]
        elif attention_mask.shape[1] < seq_len:
            padded_mask = np.zeros((batch_size, seq_len), dtype=np.float32)
            padded_mask[:, : attention_mask.shape[1]] = attention_mask
            attention_mask = padded_mask

        # Pad to max_model_len for kernel execution (kernels expect fixed size)
        if seq_len < self.config.max_model_len:
            padded_input_ids = np.zeros(
                (batch_size, self.config.max_model_len), dtype=np.uint32
            )
            padded_input_ids[:, :seq_len] = input_ids
            input_ids = padded_input_ids

            padded_mask = np.zeros(
                (batch_size, self.config.max_model_len), dtype=np.float32
            )
            padded_mask[:, :seq_len] = attention_mask
            attention_mask = padded_mask

        # STEP 1: Token embedding
        input_ids_device = DeviceTensor.from_numpy(input_ids, "input_ids")
        hidden_states_3d = DeviceTensor.from_numpy(
            np.zeros(
                (batch_size, self.config.max_model_len, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            "hidden_states_3d",
        )

        self.token_embedding_kernel(
            inputs={
                "tok_embedding": self.tok_embedding_device,
                "token_ids": input_ids_device,
            },
            outputs={f"{OUTPUT_PREFIX}0": hidden_states_3d},
        )

        # Reshape to 2D for layer processing
        hidden_states_2d = DeviceTensor.from_numpy(
            hidden_states_3d.numpy().reshape(
                batch_size * self.config.max_model_len, self.config.hidden_size
            ),
            "hidden_states_2d",
        )

        # STEP 2: Pass through all transformer layers
        # Each layer uses the same compiled kernels with different weights
        for layer in self.layers:
            hidden_states_2d = layer.forward(hidden_states_2d)

        # STEP 3: Final layer normalization
        normed_hidden_states = DeviceTensor.from_numpy(
            np.empty_like(hidden_states_2d.numpy()), "final_normed_hidden_states"
        )

        self.final_norm_kernel(
            inputs={"x": hidden_states_2d, "weight": self.norm_weight_device},
            outputs={f"{OUTPUT_PREFIX}0": normed_hidden_states},
        )

        # Reshape back to 3D
        final_hidden_states = normed_hidden_states.numpy().reshape(
            batch_size, self.config.max_model_len, self.config.hidden_size
        )

        # STEP 4: Extract embeddings using last token pooling
        embeddings = last_token_pool(final_hidden_states, attention_mask)

        return embeddings
