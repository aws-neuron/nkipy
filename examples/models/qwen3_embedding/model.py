"""Qwen3-Embedding model for Trainium with configurable kernel execution."""

from typing import Optional

import numpy as np
from config import BUILD_DIR, Qwen3Config
from embedding_utils import last_token_pool
from kernels.attention import qwen3_attention_kernel
from kernels.ffn import feedforward_kernel
from kernels.rmsnorm import rmsnorm
from kernels.rope import compute_qwen3_cos_sin
from kernels.token_embedding import token_embedding
from kernels.transformer_layer import transformer_layer_kernel
from logger import get_logger
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import DeviceKernel, DeviceTensor

logger = get_logger()

BACKEND = "hlo"
OUTPUT_PREFIX = "output"


class Qwen3EmbeddingModel:
    """Qwen3-Embedding model for Trainium.

    Features:
    - Configurable LNC (Logical NeuronCore) mode for compilation
    - Shared kernels across all transformer layers
    - Grouped Query Attention (GQA)
    - Q/K normalization before RoPE
    - Last token pooling for embedding extraction

    Args:
        weights: Dictionary of model weights
        config: Model configuration
        lnc: Logical NeuronCore count (1 or 2, default: 2)
        use_fused_layer: If True, use fused transformer layer kernel (default).
                         If False, use separate attention/ffn kernels via layer.py.
    """

    def __init__(
        self,
        weights: dict,
        config: Qwen3Config,
        lnc: int = 2,
        use_fused_layer: bool = True,
    ):
        self.config = config
        self.lnc = lnc
        self.use_fused_layer = use_fused_layer

        # Build compiler args with LNC setting
        self.additional_compiler_args = (
            f" --lnc {lnc}  --enable-mixed-precision-accumulation"
        )

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

        # Compile shared kernels
        self._compile_shared_kernels()

        # Load layer weights to device
        self._load_layer_weights(weights)

        logger.info(
            f"Initialized Qwen3 model: {config.num_hidden_layers} layers, "
            f"lnc={lnc}, fused={use_fused_layer}"
        )

    def _get_kernel_name_suffix(self) -> str:
        """Generate unique suffix for kernel names based on model config and LNC."""
        return (
            f"h{self.config.hidden_size}_v{self.config.vocab_size}_"
            f"l{self.config.num_hidden_layers}_s{self.config.max_model_len}_lnc{self.lnc}"
        )

    def _compile_shared_kernels(self):
        """Compile kernels shared across all layers.

        Fused mode (default):
        1. Token embedding
        2. Fused transformer layer (attention + FFN with residuals)
        3. Final layer norm

        Separate mode (--no-fused):
        1. Token embedding
        2. Attention kernel
        3. RMSNorm kernel (for post-attention norm)
        4. FFN kernel
        5. Final layer norm
        """
        logger.info("Compiling shared kernels...")
        kernel_suffix = self._get_kernel_name_suffix()

        # Sample tensors for kernel compilation
        hidden_states = np.empty(
            (
                self.config.max_batch_size,
                self.config.max_model_len,
                self.config.hidden_size,
            ),
            dtype=self.config.dtype,
        )

        # 1. Token embedding kernel (always needed)
        logger.info("  Compiling token embedding kernel...")
        self.token_embedding_kernel = DeviceKernel.compile_and_load(
            NKIPyKernel.trace(token_embedding, backend=BACKEND),
            name=f"qwen3_token_embedding_{kernel_suffix}",
            tok_embedding=np.empty(
                (self.config.vocab_size, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            token_ids=np.zeros(
                (self.config.max_batch_size, self.config.max_model_len), dtype=np.uint32
            ),
            build_dir=BUILD_DIR,
            additional_compiler_args=self.additional_compiler_args,
        )

        # Zero bias tensors (Qwen3 doesn't use bias)
        self.gate_up_bias = np.zeros(
            2 * self.config.intermediate_size, dtype=self.config.dtype
        )
        self.down_bias = np.zeros(self.config.hidden_size, dtype=self.config.dtype)

        qkv_size = (
            self.config.num_attention_heads + 2 * self.config.num_key_value_heads
        ) * self.config.head_dim

        if self.use_fused_layer:
            # 2. Fused transformer layer kernel
            logger.info("  Compiling fused transformer layer kernel...")
            self.shared_layer_kernel = DeviceKernel.compile_and_load(
                kernel=NKIPyKernel.trace(transformer_layer_kernel, backend=BACKEND),
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
                post_attention_layernorm_weight=np.empty(
                    self.config.hidden_size, dtype=self.config.dtype
                ),
                gate_up_weight=np.empty(
                    (self.config.hidden_size, 2 * self.config.intermediate_size),
                    dtype=self.config.dtype,
                ),
                down_weight=np.empty(
                    (self.config.intermediate_size, self.config.hidden_size),
                    dtype=self.config.dtype,
                ),
                gate_up_bias=self.gate_up_bias,
                down_bias=self.down_bias,
                config=self.config,
                compute_dtype=self.config.dtype,
                name=f"qwen3_transformer_layer_{kernel_suffix}",
                build_dir=BUILD_DIR,
                additional_compiler_args=self.additional_compiler_args,
            )
        else:
            # Separate kernels mode
            # 2a. Attention kernel
            logger.info("  Compiling attention kernel...")
            self.attention_kernel = DeviceKernel.compile_and_load(
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
                name=f"qwen3_attention_{kernel_suffix}",
                build_dir=BUILD_DIR,
                additional_compiler_args=self.additional_compiler_args,
            )

            # 2b. RMSNorm kernel (for post-attention normalization)
            logger.info("  Compiling rmsnorm kernel...")
            self.rmsnorm_kernel = DeviceKernel.compile_and_load(
                NKIPyKernel.trace(rmsnorm, backend=BACKEND),
                x=hidden_states,
                weight=np.empty(self.config.hidden_size, dtype=self.config.dtype),
                eps=self.config.rms_norm_eps,
                name=f"qwen3_rmsnorm_{kernel_suffix}",
                build_dir=BUILD_DIR,
                additional_compiler_args=self.additional_compiler_args,
            )

            # 2c. FFN kernel
            logger.info("  Compiling ffn kernel...")
            self.ffn_kernel = DeviceKernel.compile_and_load(
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
                gate_up_bias=self.gate_up_bias,
                down_bias=self.down_bias,
                name=f"qwen3_ffn_{kernel_suffix}",
                build_dir=BUILD_DIR,
                additional_compiler_args=self.additional_compiler_args,
            )

        # 3. Final layer norm kernel (always needed)
        logger.info("  Compiling final norm kernel...")
        self.final_norm_kernel = DeviceKernel.compile_and_load(
            NKIPyKernel.trace(rmsnorm, backend=BACKEND),
            x=hidden_states,
            weight=self.norm_weight_device,
            eps=self.config.rms_norm_eps,
            name=f"qwen3_final_norm_{kernel_suffix}",
            build_dir=BUILD_DIR,
            additional_compiler_args=self.additional_compiler_args,
        )

        logger.info("All shared kernels compiled successfully!")

    def _load_layer_weights(self, weights: dict):
        """Load all layer weights to device tensors."""
        self.layer_weights = []

        # Device tensors for zero biases (shared across all layers)
        self.gate_up_bias_device = DeviceTensor.from_numpy(
            self.gate_up_bias, "gate_up_bias_zero"
        )
        self.down_bias_device = DeviceTensor.from_numpy(
            self.down_bias, "down_bias_zero"
        )

        for layer_id in range(self.config.num_hidden_layers):
            prefix = f"layers.{layer_id}"
            layer_weight_dict = {
                "qkv_weight": DeviceTensor.from_torch(
                    weights.pop(f"{prefix}.qkv_weight"), f"qkv_weight_L{layer_id}"
                ),
                "q_norm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{prefix}.q_norm_weight"), f"q_norm_weight_L{layer_id}"
                ),
                "k_norm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{prefix}.k_norm_weight"), f"k_norm_weight_L{layer_id}"
                ),
                "o_weight": DeviceTensor.from_torch(
                    weights.pop(f"{prefix}.o_weight"), f"o_weight_L{layer_id}"
                ),
                "input_layernorm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{prefix}.input_layernorm_weight"),
                    f"input_layernorm_weight_L{layer_id}",
                ),
                "post_attention_layernorm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{prefix}.post_attention_layernorm_weight"),
                    f"post_attention_layernorm_weight_L{layer_id}",
                ),
                "gate_up_weight": DeviceTensor.from_torch(
                    weights.pop(f"{prefix}.gate_up_weight"),
                    f"gate_up_weight_L{layer_id}",
                ),
                "down_weight": DeviceTensor.from_torch(
                    weights.pop(f"{prefix}.down_weight"), f"down_weight_L{layer_id}"
                ),
            }
            self.layer_weights.append(layer_weight_dict)

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)

        Returns:
            embeddings: [batch_size, hidden_size]
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

        # Pad to max_model_len for kernel execution
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

        # Token embedding
        input_ids_device = DeviceTensor.from_numpy(input_ids, "input_ids")
        hidden_states_zeros = np.zeros(
            (batch_size, self.config.max_model_len, self.config.hidden_size),
            dtype=self.config.dtype,
        )
        hidden_states = DeviceTensor.from_numpy(hidden_states_zeros, "hidden/output_0")

        self.token_embedding_kernel(
            inputs={
                "tok_embedding": self.tok_embedding_device,
                "token_ids": input_ids_device,
            },
            outputs={f"{OUTPUT_PREFIX}0": hidden_states},
        )

        # Transformer layers
        layer_output = DeviceTensor.from_numpy(hidden_states_zeros, "hidden/output_1")

        if self.use_fused_layer:
            # Fused mode: single kernel per layer
            for layer_id, layer_weight in enumerate(self.layer_weights):
                self.shared_layer_kernel(
                    inputs={
                        "hidden_states": hidden_states,
                        "input_layernorm_weight": layer_weight[
                            "input_layernorm_weight"
                        ],
                        "qkv_weight": layer_weight["qkv_weight"],
                        "o_weight": layer_weight["o_weight"],
                        "q_norm_weight": layer_weight["q_norm_weight"],
                        "k_norm_weight": layer_weight["k_norm_weight"],
                        "cos": self.cos,
                        "sin": self.sin,
                        "post_attention_layernorm_weight": layer_weight[
                            "post_attention_layernorm_weight"
                        ],
                        "gate_up_weight": layer_weight["gate_up_weight"],
                        "down_weight": layer_weight["down_weight"],
                        "gate_up_bias": self.gate_up_bias_device,
                        "down_bias": self.down_bias_device,
                    },
                    outputs={f"{OUTPUT_PREFIX}0": layer_output},
                )
                # Swap tensors for next iteration
                hidden_states, layer_output = layer_output, hidden_states
        else:
            # Separate kernels mode: attention -> rmsnorm -> ffn per layer
            attn_output = DeviceTensor.from_numpy(hidden_states_zeros, "attn_output")
            normed_output = DeviceTensor.from_numpy(
                hidden_states_zeros, "normed_output"
            )
            ffn_output = DeviceTensor.from_numpy(hidden_states_zeros, "ffn_output")

            for layer_id, layer_weight in enumerate(self.layer_weights):
                # 1. Attention (includes input norm and residual)
                self.attention_kernel(
                    inputs={
                        "hidden_states": hidden_states,
                        "input_layernorm_weight": layer_weight[
                            "input_layernorm_weight"
                        ],
                        "qkv_weight": layer_weight["qkv_weight"],
                        "o_weight": layer_weight["o_weight"],
                        "q_norm_weight": layer_weight["q_norm_weight"],
                        "k_norm_weight": layer_weight["k_norm_weight"],
                        "cos": self.cos,
                        "sin": self.sin,
                    },
                    outputs={f"{OUTPUT_PREFIX}0": attn_output},
                )

                # 2. Post-attention RMSNorm
                self.rmsnorm_kernel(
                    inputs={
                        "x": attn_output,
                        "weight": layer_weight["post_attention_layernorm_weight"],
                    },
                    outputs={f"{OUTPUT_PREFIX}0": normed_output},
                )

                # 3. FFN
                self.ffn_kernel(
                    inputs={
                        "x": normed_output,
                        "gate_up_weight": layer_weight["gate_up_weight"],
                        "down_weight": layer_weight["down_weight"],
                        "gate_up_bias": self.gate_up_bias_device,
                        "down_bias": self.down_bias_device,
                    },
                    outputs={f"{OUTPUT_PREFIX}0": ffn_output},
                )

                # 4. Add residual (FFN output + attention output)
                ffn_np = ffn_output.numpy()
                attn_np = attn_output.numpy()
                hidden_states = DeviceTensor.from_numpy(
                    ffn_np + attn_np, f"hidden_L{layer_id}"
                )

        # Final layer normalization
        normed_hidden_states = DeviceTensor.from_numpy(
            hidden_states_zeros, "final_normed_hidden_states"
        )

        self.final_norm_kernel(
            inputs={"x": hidden_states, "weight": self.norm_weight_device},
            outputs={f"{OUTPUT_PREFIX}0": normed_hidden_states},
        )

        final_hidden_states = normed_hidden_states.numpy()
        embeddings = last_token_pool(final_hidden_states, attention_mask)

        return embeddings
