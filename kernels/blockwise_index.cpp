#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <numeric>
#include <iostream>
#include <cassert>

namespace py = pybind11;

#ifndef N_EXPERTS
#error "You must define N_EXPERTS"
#endif

#ifndef TOP_K
#error "You must define TOP_K"
#endif

#ifndef N_BLOCKS
#error "You must define N_BLOCKS"
#endif

#ifndef N_STATIC_BLOCKS
#error "You must define N_STATIC_BLOCKS"
#endif

#ifndef BLOCK_SIZE
#error "You must define BLOCK_SIZE"
#endif

#ifndef SKIP_DMA
#error "You must define SKIP_DMA"
#endif

#ifndef SKIP_BLOCK
#error "You must define SKIP_BLOCK"
#endif


std::tuple<uint16_t, py::array_t<int8_t>, py::array_t<int32_t>>
get_blockwise_expert_and_token_mapping(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> top_k_indices // value out of interest masked as -1
) {
    /*
    Calculate blockwise expert and token mapping
    */
    // auto start_time = std::chrono::high_resolution_clock::now();
    auto buf = top_k_indices.request();
    const auto* ptr = static_cast<int8_t*>(buf.ptr);
    int num_tokens = buf.shape[0];
    
    // Count tokens per expert
    std::vector<uint32_t> tokens_per_expert(N_EXPERTS, 0);
    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < TOP_K; k++) {
            auto e = ptr[t * TOP_K + k];
            if (e >= 0) {
                assert(e < N_EXPERTS);
                tokens_per_expert[e]++;
            }
        }
    }

    std::vector<uint16_t> blocks_per_expert(N_EXPERTS);
    for (int e = 0; e < N_EXPERTS; e++) {
        blocks_per_expert[e] = (tokens_per_expert[e] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
    
    std::vector<uint16_t> start_block_per_expert(N_EXPERTS);
    uint16_t num_real_blocks = 0;    
    for (int e = 0; e < N_EXPERTS; ++e) { 
      start_block_per_expert[e] = num_real_blocks; 
      num_real_blocks += blocks_per_expert[e];
    }
    
    // Check for potential underflow
    if (num_real_blocks > N_BLOCKS) {
        std::cerr << "Error: num_real_blocks (" << num_real_blocks << ") > N_BLOCKS (" << N_BLOCKS << ")" << std::endl;
        throw std::runtime_error("num_real_blocks exceeds N_BLOCKS - possible configuration error");
    }
    
    // Allocate numpy arrays directly and write to them
    auto result_block_to_expert = py::array_t<int8_t>(N_BLOCKS);
    auto block_to_expert = result_block_to_expert.mutable_unchecked<1>();
    
    // Initialize all to SKIP_BLOCK
    for (int i = 0; i < N_BLOCKS; ++i) {
        block_to_expert(i) = SKIP_BLOCK;
    }
    
    uint16_t current_block_idx = 0;
    for (int e = 0; e < N_EXPERTS; ++e) {
      for (uint16_t b = 0; b < blocks_per_expert[e]; ++b) {
        block_to_expert(current_block_idx++) = e;
      }
    }
    // reverse order to avoid write affect read
    if (num_real_blocks > 1) {  // Guard against underflow when num_real_blocks is 0 or 1
      for (uint16_t b = num_real_blocks - 1; b > 0; --b) {
        if (block_to_expert(b) == block_to_expert(b-1) && b != N_STATIC_BLOCKS) {
          block_to_expert(b) = SKIP_DMA;
        }
      }
    }
    
    std::vector<py::ssize_t> shape = {N_BLOCKS, BLOCK_SIZE};
    auto result_token_position_to_id = py::array_t<int32_t>(shape);
    auto token_position_to_id = result_token_position_to_id.mutable_unchecked<2>();
    
    // Initialize all to SKIP_DMA
    for (int i = 0; i < N_BLOCKS; ++i) {
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            token_position_to_id(i, j) = SKIP_DMA;
        }
    }
    
    std::vector<int32_t> current_token_per_expert(N_EXPERTS, 0);
    for (int t = 0; t < num_tokens; ++t) {
      for (int k = 0; k < TOP_K; ++k) {
        auto e = ptr[t*TOP_K + k];
        if (e >= 0) {
          uint32_t pos = start_block_per_expert[e] * (uint32_t)BLOCK_SIZE + current_token_per_expert[e]++;
          uint32_t block_idx = pos / BLOCK_SIZE;
          uint32_t within_block = pos % BLOCK_SIZE;
          token_position_to_id(block_idx, within_block) = t;
        }
      }
    }

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // double elapsed_time = duration.count() / 1000.0; // Convert to milliseconds
    // std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;
    return std::make_tuple(num_real_blocks, result_block_to_expert, result_token_position_to_id);
}

PYBIND11_MODULE(blockwise_index_cpp, m)
{
    m.def("get_blockwise_expert_and_token_mapping", &get_blockwise_expert_and_token_mapping,
          "C++ implementation of the blockwise mapping function",
          py::arg("top_k_indices"));
}
