// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
// blockwise_index.cpp
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace {

constexpr int32_t SKIP_DMA = -1;
constexpr int32_t SKIP_BLOCK = -2;

// top_k_indices: [T, TOP_K], dtype=int8 or int32; we'll make it int32 inside
// Returns: (num_real_blocks: int, block_to_expert: int8[N], token_position_to_id: int32[N, B])
std::tuple<int64_t, at::Tensor, at::Tensor>
get_blockwise_expert_and_token_mapping(
    const at::Tensor& top_k_indices,
    int64_t num_blocks,
    int64_t block_size,
    int64_t num_experts,
    int64_t num_static_blocks
) {
    TORCH_CHECK(top_k_indices.dim() == 2, "top_k_indices must be [T, TOP_K]");
    TORCH_CHECK(top_k_indices.is_contiguous(), "top_k_indices must be contiguous");
    TORCH_CHECK(top_k_indices.device().is_cpu(), "this example only supports CPU for now");

    auto T = top_k_indices.size(0);
    auto TOP_K = top_k_indices.size(1);

    // We'll work in int32 view no matter what was passed
    at::Tensor topk_i32;
    if (top_k_indices.scalar_type() == at::kInt) {
        topk_i32 = top_k_indices;
    } else {
        // e.g. original was int8
        topk_i32 = top_k_indices.to(at::kInt);
    }

    // 1) count tokens per expert
    std::vector<int64_t> tokens_per_expert(num_experts, 0);
    auto topk_ptr = topk_i32.data_ptr<int32_t>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < TOP_K; ++k) {
            int32_t e = topk_ptr[t * TOP_K + k];
            if (e >= 0) {
                TORCH_CHECK(e < num_experts, "expert id out of range");
                tokens_per_expert[e] += 1;
            }
        }
    }

    // 2) blocks per expert
    std::vector<int64_t> blocks_per_expert(num_experts, 0);
    for (int64_t e = 0; e < num_experts; ++e) {
        // ceil(tokens / block_size)
        blocks_per_expert[e] = (tokens_per_expert[e] + block_size - 1) / block_size;
    }

    // 3) prefix to know starting block of each expert
    std::vector<int64_t> start_block_per_expert(num_experts, 0);
    int64_t num_real_blocks = 0;
    for (int64_t e = 0; e < num_experts; ++e) {
        start_block_per_expert[e] = num_real_blocks;
        num_real_blocks += blocks_per_expert[e];
    }

    TORCH_CHECK(
        num_real_blocks <= num_blocks,
        "num_real_blocks (", num_real_blocks, ") > num_blocks (", num_blocks, ")"
    );

    // 4) allocate outputs
    // block_to_expert: int8[num_blocks], init to SKIP_BLOCK
    auto block_to_expert = at::empty({num_blocks}, at::TensorOptions().dtype(at::kChar));  // int8
    auto block_to_expert_ptr = block_to_expert.data_ptr<int8_t>();
    for (int64_t i = 0; i < num_blocks; ++i) {
        block_to_expert_ptr[i] = static_cast<int8_t>(SKIP_BLOCK);
    }

    // fill the real blocks
    int64_t cur_block = 0;
    for (int64_t e = 0; e < num_experts; ++e) {
        for (int64_t b = 0; b < blocks_per_expert[e]; ++b) {
            block_to_expert_ptr[cur_block++] = static_cast<int8_t>(e);
        }
    }

    // mark consecutive blocks of same expert as SKIP_DMA, except the ones <= num_static_blocks
    if (num_real_blocks > 1) {
        for (int64_t b = num_real_blocks - 1; b > 0; --b) {
            if (block_to_expert_ptr[b] == block_to_expert_ptr[b - 1] &&
                b != num_static_blocks) {
                block_to_expert_ptr[b] = static_cast<int8_t>(SKIP_DMA);
            }
        }
    }

    // 5) token_position_to_id: [num_blocks, block_size], int32, init SKIP_DMA
    auto token_position_to_id = at::empty({num_blocks, block_size},
                                          at::TensorOptions().dtype(at::kInt));
    auto token_pos_ptr = token_position_to_id.data_ptr<int32_t>();
    for (int64_t b = 0; b < num_blocks; ++b) {
        for (int64_t i = 0; i < block_size; ++i) {
            token_pos_ptr[b * block_size + i] = SKIP_DMA;
        }
    }

    // track how many tokens we've placed for each expert
    std::vector<int64_t> current_token_per_expert(num_experts, 0);

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < TOP_K; ++k) {
            int32_t e = topk_ptr[t * TOP_K + k];
            if (e >= 0) {
                int64_t global_pos =
                    start_block_per_expert[e] * block_size + current_token_per_expert[e];
                int64_t block_idx = global_pos / block_size;
                int64_t within_block = global_pos % block_size;
                token_pos_ptr[block_idx * block_size + within_block] = static_cast<int32_t>(t);
                current_token_per_expert[e] += 1;
            }
        }
    }

    return std::make_tuple(num_real_blocks, block_to_expert, token_position_to_id);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "get_blockwise_expert_and_token_mapping",
        &get_blockwise_expert_and_token_mapping,
        pybind11::arg("top_k_indices"),
        pybind11::arg("num_blocks"),
        pybind11::arg("block_size"),
        pybind11::arg("num_experts"),
        pybind11::arg("num_static_blocks"),
        "C++/Torch implementation of blockwise mapping (single compile, runtime sizes)"
    );
}
