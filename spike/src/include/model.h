// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef SPIKE_SRC_INCLUDE_MODEL_H
#define SPIKE_SRC_INCLUDE_MODEL_H

#include "tensor_set.h"
#include <optional>

namespace spike {

class Spike;

// RAII wrapper for NRT model
class NrtModel {
public:
  // This constructor creates an NrtModel that owns the model
  NrtModel(const std::string &neff_path, uint32_t core_id, bool cc_enabled,
           uint32_t rank_id, uint32_t world_size, const Spike *spike);
  // This constructor creates an NrtModel that references an existing model
  NrtModel(nrt_model_t *ptr, uint32_t core_id, bool cc_enabled,
           uint32_t rank_id, uint32_t world_size);
  ~NrtModel();

  // Non-copyable, movable
  NrtModel(const NrtModel &) = delete;
  NrtModel &operator=(const NrtModel &) = delete;
  NrtModel(NrtModel &&other) noexcept;
  NrtModel &operator=(NrtModel &&other) noexcept;

  // Getters (read-only properties for Python)
  nrt_model_t *get_ptr() const { return ptr_; }
  const std::string &get_neff_path() const { return neff_path_; }
  uint32_t get_core_id() const { return core_id_; }
  uint32_t get_rank_id() const { return rank_id_; }
  uint32_t get_world_size() const { return world_size_; }
  bool get_is_collective() const { return is_collective_; }
  bool is_unloaded() const;
  bool is_owner() const { return spike_ != nullptr; }

  // String representation
  std::string to_string() const;

  void execute(const NrtTensorSet &inputs, NrtTensorSet &outputs,
               std::optional<std::string> ntff_name, bool save_trace);
  nrt_tensor_info_array_t *get_tensor_info();
  static void free_tensor_info(nrt_tensor_info_array_t *info);
  void unload();

private:
  nrt_model_t *ptr_;
  std::string neff_path_;
  uint32_t core_id_;
  uint32_t rank_id_;
  uint32_t world_size_;
  bool is_collective_;
  const Spike *spike_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_MODEL_H
