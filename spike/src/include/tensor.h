// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef SPIKE_SRC_INCLUDE_TENSOR_H
#define SPIKE_SRC_INCLUDE_TENSOR_H

#include "nrt_wrapper.h"

#include <cstdint>
#include <string>

namespace spike {

class Spike;

// RAII wrapper for NRT tensor
class NrtTensor {
public:
  // This constructor creates an NrtTensor that owns the tensor
  NrtTensor(nrt_tensor_placement_t placement, uint32_t core_id, size_t size,
            const std::string &name, const Spike *spike);
  NrtTensor(const NrtTensor &source, size_t offset, size_t size,
            const std::string &name);
  ~NrtTensor();

  // Getters (read-only properties for Python)
  nrt_tensor_t *get_ptr() const { return ptr_; }
  uint32_t get_core_id() const { return core_id_; }
  uint64_t get_size() const { return size_; }
  const std::string &get_name() const { return name_; }
  bool is_freed() const;
  bool is_owner() const { return spike_ != nullptr; }

  // String representation
  std::string to_string() const;

  // Non-copyable, movable
  NrtTensor(const NrtTensor &) = delete;
  NrtTensor &operator=(const NrtTensor &) = delete;
  NrtTensor(NrtTensor &&other) noexcept;
  NrtTensor &operator=(NrtTensor &&other) noexcept;

  void write(const void *data, size_t size, size_t offset = 0);
  void read(void *data, size_t size, size_t offset = 0) const;
  void free();

private:
  nrt_tensor_t *ptr_;
  uint32_t core_id_;
  uint64_t size_;
  std::string name_;
  const Spike *spike_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_TENSOR_H
