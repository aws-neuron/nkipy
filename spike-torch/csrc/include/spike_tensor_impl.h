// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <nrt/nrt.h>

namespace spike_torch {

// Custom TensorImpl for spike tensors
// Sets appropriate dispatch keys for PrivateUse1 device
class SpikeTensorImpl : public c10::TensorImpl {
public:
  explicit SpikeTensorImpl(c10::Storage &&storage,
                           const caffe2::TypeMeta &data_type);

  // Shallow copy operations
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl) override;
  c10::intrusive_ptr<c10::TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const override;
  c10::intrusive_ptr<c10::TensorImpl>
  shallow_copy_and_detach(c10::VariableVersion &&version_counter,
                          bool allow_tensor_metadata_change) const override;

private:
  // Helper to set device-specific dispatch keys
  void SetSpikeDispatchKeys();
};

// Tensor operations registered with PyTorch dispatch
namespace ops {

// Empty tensor creation
at::Tensor empty_memory_format(c10::IntArrayRef size,
                               c10::optional<at::ScalarType> dtype_opt,
                               c10::optional<c10::Layout> layout_opt,
                               c10::optional<c10::Device> device_opt,
                               c10::optional<bool> pin_memory_opt,
                               c10::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         c10::optional<c10::ScalarType> dtype_opt,
                         c10::optional<c10::Layout> layout_opt,
                         c10::optional<c10::Device> device_opt,
                         c10::optional<bool> pin_memory_opt);

// Resize operations
const at::Tensor &resize_(const at::Tensor &self, c10::IntArrayRef size,
                          std::optional<c10::MemoryFormat> memory_format);

// Copy operations
at::Tensor &copy_(at::Tensor &self, const at::Tensor &src,
                  bool non_blocking = false);

at::Tensor _to_copy(const at::Tensor &self,
                    c10::optional<at::ScalarType> dtype,
                    c10::optional<c10::Layout> layout,
                    c10::optional<c10::Device> device,
                    c10::optional<bool> pin_memory, bool non_blocking,
                    c10::optional<c10::MemoryFormat> memory_format);

at::Tensor _copy_from_and_resize(const at::Tensor &self, const at::Tensor &dst);

// View operations
at::Tensor view(const at::Tensor &self, at::IntArrayRef size);

at::Tensor as_strided(const at::Tensor &self, at::IntArrayRef size,
                      at::IntArrayRef stride,
                      c10::optional<int64_t> storage_offset);

at::Tensor _reshape_alias(const at::Tensor &self, at::IntArrayRef sizes,
                          at::IntArrayRef strides);

// Fill operations
at::Tensor &fill_scalar(at::Tensor &self, const at::Scalar &value);

} // namespace ops

// Helper to get the NRT tensor from a PyTorch tensor
nrt_tensor_t *get_nrt_tensor(const at::Tensor &tensor);

// Helper to create spike storage
c10::Storage make_spike_storage(size_t size_bytes, int device_index);

} // namespace spike_torch
