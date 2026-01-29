// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "include/spike_tensor_impl.h"

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Resize.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <nrt/nrt.h>

#include <algorithm>

#include "include/spike_allocator.h"
#include "include/spike_device.h"
#include "include/spike_storage_impl.h"

namespace spike_torch {

SpikeTensorImpl::SpikeTensorImpl(c10::Storage &&storage,
                                 const caffe2::TypeMeta &data_type)
    : c10::TensorImpl(std::move(storage),
                      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                      data_type) {
  SetSpikeDispatchKeys();
}

void SpikeTensorImpl::SetSpikeDispatchKeys() {
  key_set_ = key_set_.add(c10::DispatchKey::PrivateUse1);
  key_set_ = key_set_.add(c10::DispatchKey::AutogradPrivateUse1);
}

void SpikeTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl> &impl) {
  TensorImpl::shallow_copy_from(impl);
  SetSpikeDispatchKeys();
}

c10::intrusive_ptr<c10::TensorImpl> SpikeTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion &version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl =
      c10::make_intrusive<SpikeTensorImpl>(c10::Storage(storage()), data_type_);

  copy_tensor_metadata(this, impl.get(), version_counter,
                       allow_tensor_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> SpikeTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion &&version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl =
      c10::make_intrusive<SpikeTensorImpl>(c10::Storage(storage()), data_type_);

  copy_tensor_metadata(this, impl.get(), std::move(version_counter),
                       allow_tensor_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

// Helper to create spike storage
c10::Storage make_spike_storage(size_t size_bytes, int device_index) {
  c10::Allocator *alloc = allocator::get();
  device::set_device(device_index);
  auto data_ptr = alloc->allocate(size_bytes);

  auto storage_impl = c10::make_intrusive<SpikeStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr),
      alloc, true);

  return c10::Storage(std::move(storage_impl));
}

// Get NRT tensor from PyTorch tensor
nrt_tensor_t *get_nrt_tensor(const at::Tensor &tensor) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              "Expected spike tensor, got ", tensor.device());

  auto storage_impl = tensor.storage().unsafeGetStorageImpl();
  auto spike_storage = static_cast<SpikeStorageImpl *>(storage_impl);
  return spike_storage->nrt_tensor();
}

namespace ops {

at::Tensor empty_memory_format(c10::IntArrayRef size,
                               c10::optional<at::ScalarType> dtype_opt,
                               c10::optional<c10::Layout> layout_opt,
                               c10::optional<c10::Device> device_opt,
                               c10::optional<bool> pin_memory_opt,
                               c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto memory_format =
      memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous,
              "Spike backend only supports contiguous memory format, but got: ",
              memory_format);
  return empty_strided(size, c10::TensorType::contiguousStridesOf(size),
                       dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         c10::optional<c10::ScalarType> dtype_opt,
                         c10::optional<c10::Layout> layout_opt,
                         c10::optional<c10::Device> device_opt,
                         c10::optional<bool> pin_memory_opt) {
  auto dev = c10::device_or_default(device_opt);
  TORCH_CHECK(dev.type() == c10::DeviceType::PrivateUse1,
              "Expected spike device but got: ", dev);

  int device_index = dev.has_index() ? dev.index() : 0;
  auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));

  int64_t nelements = c10::multiply_integers(size);
  int64_t size_bytes = nelements * dtype.itemsize();

  // Handle zero-size tensors
  if (size_bytes == 0) {
    size_bytes = dtype.itemsize();
  }

  auto storage = make_spike_storage(size_bytes, device_index);
  auto tensor =
      at::detail::make_tensor<SpikeTensorImpl>(std::move(storage), dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

const at::Tensor &resize_(const at::Tensor &self, c10::IntArrayRef size,
                          std::optional<c10::MemoryFormat> memory_format) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "resize_: expected spike tensor, but got: ", self.device());

  auto format = memory_format.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(format == c10::MemoryFormat::Contiguous ||
                  format == c10::MemoryFormat::Preserve,
              "Spike backend only supports contiguous or preserve memory "
              "format, but got: ",
              format);

  if (format == c10::MemoryFormat::Preserve) {
    format = c10::MemoryFormat::Contiguous;
  }

  int64_t new_nelements = c10::multiply_integers(size);
  auto dtype = self.scalar_type();
  size_t itemsize = c10::elementSize(dtype);
  size_t new_size_bytes = new_nelements * itemsize;
  size_t current_size = self.storage().nbytes();

  if (new_size_bytes > current_size) {
    int device_index = self.device().index();
    device::set_device(device_index);
    auto new_storage = make_spike_storage(new_size_bytes, device_index);

    if (self.numel() > 0 && current_size > 0) {
      void *old_data = self.data_ptr();
      void *new_data = new_storage.data_ptr().get();
      size_t copy_size = std::min(current_size, new_size_bytes);
      bool success = allocator::copy_tensor_data(new_data, old_data, copy_size);
      TORCH_CHECK(success, "Failed to copy data during resize");
    }

    self.unsafeGetTensorImpl()->set_storage_and_dtype(
        new_storage, c10::scalarTypeToTypeMeta(dtype));
  }

  if (new_nelements > 0) {
    self.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  } else {
    std::vector<int64_t> strides(size.size());
    int64_t stride = 1;
    for (int64_t i = size.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= size[i];
    }
    self.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);
  }

  return self;
}

// Copy helpers
namespace {

void copy_cpu_to_spike(const at::Tensor &src, at::Tensor &dst) {
  TORCH_CHECK(src.is_cpu(), "Source tensor must be CPU tensor");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "Destination must be spike tensor");
  TORCH_CHECK(src.is_contiguous(),
              "Source CPU tensor must be contiguous for copy to spike");
  TORCH_CHECK(dst.is_contiguous(), "Destination spike tensor must be contiguous");

  nrt_tensor_t *nrt_dst = get_nrt_tensor(dst);
  TORCH_CHECK(nrt_dst != nullptr, "Failed to get spike tensor handle");

  size_t byte_offset = dst.storage_offset() * dst.element_size();

  if (src.nbytes() == 0) {
    return;
  }

  NRT_STATUS status =
      nrt_tensor_write(nrt_dst, src.data_ptr(), byte_offset, src.nbytes());
  TORCH_CHECK(status == NRT_SUCCESS,
              "Failed to copy data to spike device. Status: ", status);
}

void copy_spike_to_cpu(const at::Tensor &src, at::Tensor &dst) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "Source must be spike tensor");
  TORCH_CHECK(dst.is_cpu(), "Destination tensor must be CPU tensor");

  if (!src.is_contiguous() && dst.is_contiguous()) {
    TORCH_CHECK(false, "Source spike tensor must be contiguous for copy to CPU");
  }

  TORCH_CHECK(dst.is_contiguous(),
              "Destination CPU tensor must be contiguous for copy from spike");

  nrt_tensor_t *nrt_src = get_nrt_tensor(src);
  TORCH_CHECK(nrt_src != nullptr, "Failed to get spike tensor handle");

  size_t byte_offset = src.storage_offset() * src.element_size();

  if (dst.nbytes() == 0) {
    return;
  }

  NRT_STATUS status =
      nrt_tensor_read(nrt_src, dst.data_ptr(), byte_offset, dst.nbytes());
  TORCH_CHECK(status == NRT_SUCCESS,
              "Failed to copy data from spike device. Status: ", status);
}

void copy_spike_to_spike(const at::Tensor &src, at::Tensor &dst) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "Source must be spike tensor");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "Destination must be spike tensor");
  TORCH_CHECK(src.device().index() == dst.device().index(),
              "Device-to-device copy requires tensors on same Neuron core");

  if (!src.is_contiguous() && dst.is_contiguous()) {
    TORCH_CHECK(false, "Source spike tensor must be contiguous for D2D copy");
  }

  TORCH_CHECK(dst.is_contiguous(), "Destination spike tensor must be contiguous");

  nrt_tensor_t *nrt_src = get_nrt_tensor(src);
  nrt_tensor_t *nrt_dst = get_nrt_tensor(dst);
  TORCH_CHECK(nrt_src != nullptr, "Failed to get source spike tensor handle");
  TORCH_CHECK(nrt_dst != nullptr, "Failed to get destination spike tensor handle");

  size_t src_byte_offset = src.storage_offset() * src.element_size();
  size_t dst_byte_offset = dst.storage_offset() * dst.element_size();

  TORCH_CHECK(src.nbytes() == dst.nbytes(),
              "Source and destination must have same number of bytes");

  if (src.nbytes() == 0) {
    return;
  }

  NRT_STATUS status = nrt_tensor_copy(nrt_src, src_byte_offset, nrt_dst,
                                      dst_byte_offset, src.nbytes());
  TORCH_CHECK(status == NRT_SUCCESS,
              "Failed to copy between spike tensors. Status: ", status);
}

} // namespace

at::Tensor &copy_(at::Tensor &self, const at::Tensor &src, bool non_blocking) {
  if (self.is_same(src)) {
    return self;
  }

  TORCH_CHECK(self.sizes() == src.sizes(),
              "Source and destination sizes must match. Got src: ", src.sizes(),
              " and dst: ", self.sizes());

  bool same_type = self.dtype() == src.dtype();

  // CPU to spike
  if (src.is_cpu() && self.device().type() == c10::DeviceType::PrivateUse1) {
    if (!same_type) {
      at::Tensor src_converted = src.to(self.dtype());
      copy_cpu_to_spike(src_converted, self);
    } else {
      copy_cpu_to_spike(src, self);
    }
    return self;
  }

  // spike to CPU
  if (src.device().type() == c10::DeviceType::PrivateUse1 && self.is_cpu()) {
    if (!same_type) {
      at::Tensor temp_cpu =
          at::empty(src.sizes(), self.options().dtype(src.dtype()));
      copy_spike_to_cpu(src, temp_cpu);
      self.copy_(temp_cpu);
    } else {
      copy_spike_to_cpu(src, self);
    }
    return self;
  }

  // spike to spike
  if (src.device().type() == c10::DeviceType::PrivateUse1 &&
      self.device().type() == c10::DeviceType::PrivateUse1) {
    if (!same_type) {
      at::Tensor temp_cpu = src.to(at::kCPU, src.dtype());
      temp_cpu = temp_cpu.to(self.dtype());
      copy_cpu_to_spike(temp_cpu, self);
    } else {
      copy_spike_to_spike(src, self);
    }
    return self;
  }

  TORCH_CHECK(false, "Unsupported device combination for copy_");
}

at::Tensor _to_copy(const at::Tensor &self,
                    c10::optional<at::ScalarType> dtype,
                    c10::optional<c10::Layout> layout,
                    c10::optional<c10::Device> device,
                    c10::optional<bool> pin_memory, bool non_blocking,
                    c10::optional<c10::MemoryFormat> memory_format) {
  bool is_to_spike =
      device.has_value() && device->type() == c10::DeviceType::PrivateUse1;
  bool is_from_spike = self.device().type() == c10::DeviceType::PrivateUse1;

  if ((is_to_spike || is_from_spike) && memory_format.has_value() &&
      *memory_format != c10::MemoryFormat::Contiguous &&
      *memory_format != c10::MemoryFormat::Preserve) {
    TORCH_CHECK(false, "Spike tensors only support contiguous memory format");
  }

  auto options = self.options();
  if (dtype.has_value())
    options = options.dtype(*dtype);
  if (layout.has_value())
    options = options.layout(*layout);
  if (device.has_value())
    options = options.device(*device);
  if (pin_memory.has_value())
    options = options.pinned_memory(*pin_memory);

  bool needs_copy = false;
  if (dtype.has_value() && *dtype != self.dtype())
    needs_copy = true;
  if (device.has_value() && *device != self.device())
    needs_copy = true;
  if (layout.has_value() && *layout != self.layout())
    needs_copy = true;

  if (!needs_copy && (!memory_format.has_value() ||
                      *memory_format == c10::MemoryFormat::Preserve)) {
    return self;
  }

  if (is_to_spike && !self.is_contiguous()) {
    at::Tensor cpu_contig = self.contiguous();
    at::Tensor result = at::empty(cpu_contig.sizes(), options);
    result.copy_(cpu_contig, non_blocking);
    return result;
  }

  at::Tensor result = at::empty(self.sizes(), options);
  result.copy_(self, non_blocking);
  return result;
}

at::Tensor _copy_from_and_resize(const at::Tensor &self,
                                 const at::Tensor &dst) {
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(self.defined(), "self is undefined");

  TORCH_CHECK(self.is_cpu() &&
                  dst.device().type() == c10::DeviceType::PrivateUse1,
              "_copy_from_and_resize only supports CPU to spike, but got src: ",
              self.device(), " and dst: ", dst.device());

  if (dst.numel() == 0) {
    const_cast<at::Tensor &>(dst).resize_as_(self);
  }

  TORCH_CHECK(self.sizes() == dst.sizes(),
              "_copy_from_and_resize requires same sizes or dst.numel() == 0");

  const_cast<at::Tensor &>(dst).copy_(self);
  return dst;
}

// Helper function to create a view tensor
static at::Tensor alias_with_sizes_and_strides(const at::Tensor &self,
                                               at::IntArrayRef sizes,
                                               at::IntArrayRef strides) {
  at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());

  auto *impl = result.unsafeGetTensorImpl();
  impl->set_storage_offset(self.storage_offset());
  impl->set_sizes_and_strides(sizes, strides);
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor view(const at::Tensor &self, at::IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);

  TORCH_CHECK(stride.has_value(),
              "view size is not compatible with input tensor's size and stride "
              "(at least one dimension spans across two contiguous subspaces). "
              "Use .reshape(...) instead.");

  return alias_with_sizes_and_strides(self, inferred_size, *stride);
}

at::Tensor as_strided(const at::Tensor &self, at::IntArrayRef size,
                      at::IntArrayRef stride,
                      c10::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());

  at::native::checkInBoundsForStorage(size, stride, storage_offset,
                                      self.dtype(), self.storage());

  at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());

  at::native::setStrided(result, size, stride, storage_offset);
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor _reshape_alias(const at::Tensor &self, at::IntArrayRef sizes,
                          at::IntArrayRef strides) {
  return view(self, sizes);
}

at::Tensor &fill_scalar(at::Tensor &self, const at::Scalar &value) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "fill_: expected spike tensor, but got: ", self.device());

  at::Tensor cpu_tensor = at::empty_like(self, self.options().device(at::kCPU));
  cpu_tensor.fill_(value);
  self.copy_(cpu_tensor);
  return self;
}

} // namespace ops
} // namespace spike_torch
