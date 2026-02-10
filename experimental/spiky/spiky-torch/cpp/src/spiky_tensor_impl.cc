#include "spiky_torch/tensor_impl.h"

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Resize.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <algorithm>

#include "spiky_torch/allocator.h"
#include "spiky_torch/device.h"
#include "spiky_torch/storage_impl.h"

extern "C" {
#include <nrt/nrt.h>
}

namespace spiky_torch {

SpikyTensorImpl::SpikyTensorImpl(c10::Storage&& storage,
                                 const caffe2::TypeMeta& data_type)
    : c10::TensorImpl(std::move(storage),
                      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                      data_type) {
  SetSpikyDispatchKeys();
}

void SpikyTensorImpl::SetSpikyDispatchKeys() {
  key_set_ = key_set_.add(c10::DispatchKey::PrivateUse1);
  key_set_ = key_set_.add(c10::DispatchKey::AutogradPrivateUse1);
}

void SpikyTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  TensorImpl::shallow_copy_from(impl);
  SetSpikyDispatchKeys();
}

c10::intrusive_ptr<c10::TensorImpl> SpikyTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<SpikyTensorImpl>(c10::Storage(storage()), data_type_);
  copy_tensor_metadata(this, impl.get(), version_counter, allow_tensor_metadata_change);
  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> SpikyTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<SpikyTensorImpl>(c10::Storage(storage()), data_type_);
  copy_tensor_metadata(this, impl.get(), std::move(version_counter), allow_tensor_metadata_change);
  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

static c10::Storage make_spiky_storage(size_t size_bytes, int device_index) {
  c10::Allocator* alloc = allocator::get();
  device::set_device(device_index);
  auto data_ptr = alloc->allocate(size_bytes);
  auto storage_impl = c10::make_intrusive<SpikyStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr), alloc, true);
  return c10::Storage(std::move(storage_impl));
}

static nrt_tensor_t* get_nrt_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              "Expected spiky tensor, got ", tensor.device());
  auto storage_impl = tensor.storage().unsafeGetStorageImpl();
  auto* spiky_storage = static_cast<SpikyStorageImpl*>(storage_impl);
  return spiky_storage->nrt_tensor();
}

namespace ops {

at::Tensor empty_memory_format(c10::IntArrayRef size,
                               c10::optional<at::ScalarType> dtype_opt,
                               c10::optional<c10::Layout> layout_opt,
                               c10::optional<c10::Device> device_opt,
                               c10::optional<bool> pin_memory_opt,
                               c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous,
              "Spiky backend only supports contiguous memory format");
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
              "Expected nkipy device but got: ", dev);
  int device_index = dev.has_index() ? dev.index() : 0;
  auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));

  int64_t nelements = c10::multiply_integers(size);
  int64_t size_bytes = nelements * dtype.itemsize();
  if (size_bytes == 0) size_bytes = dtype.itemsize();

  auto storage = make_spiky_storage(static_cast<size_t>(size_bytes), device_index);
  auto tensor = at::detail::make_tensor<SpikyTensorImpl>(std::move(storage), dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

const at::Tensor& resize_(const at::Tensor& self, c10::IntArrayRef size,
                          std::optional<c10::MemoryFormat> memory_format) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "resize_: expected nkipy tensor, but got: ", self.device());

  auto format = memory_format.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(format == c10::MemoryFormat::Contiguous ||
                  format == c10::MemoryFormat::Preserve,
              "Spiky backend only supports contiguous or preserve memory format");
  if (format == c10::MemoryFormat::Preserve) {
    format = c10::MemoryFormat::Contiguous;
  }

  int64_t new_nelements = c10::multiply_integers(size);
  auto dtype = self.scalar_type();
  size_t itemsize = c10::elementSize(dtype);
  size_t new_size_bytes = static_cast<size_t>(new_nelements) * itemsize;
  size_t current_size = self.storage().nbytes();

  if (new_size_bytes > current_size) {
    int device_index = self.device().index();
    device::set_device(device_index);
    auto new_storage = make_spiky_storage(new_size_bytes, device_index);

    if (self.numel() > 0 && current_size > 0) {
      void* old_data = self.data_ptr();
      void* new_data = new_storage.data_ptr().get();
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
    for (int64_t i = static_cast<int64_t>(size.size()) - 1; i >= 0; --i) {
      strides[static_cast<size_t>(i)] = stride;
      stride *= size[static_cast<size_t>(i)];
    }
    self.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);
  }

  return self;
}

namespace {

void copy_cpu_to_spiky(const at::Tensor& src, at::Tensor& dst) {
  TORCH_CHECK(src.is_cpu(), "Source tensor must be CPU");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1, "Destination must be nkipy");
  TORCH_CHECK(src.is_contiguous(), "Source CPU tensor must be contiguous");
  TORCH_CHECK(dst.is_contiguous(), "Destination nkipy tensor must be contiguous");

  nrt_tensor_t* nrt_dst = get_nrt_tensor(dst);
  TORCH_CHECK(nrt_dst != nullptr, "Failed to get nkipy tensor handle");
  size_t byte_offset = static_cast<size_t>(dst.storage_offset()) * dst.element_size();
  if (src.nbytes() == 0) return;
  NRT_STATUS status = nrt_tensor_write(nrt_dst, src.data_ptr(), byte_offset, src.nbytes());
  TORCH_CHECK(status == NRT_SUCCESS, "Failed CPU->nkipy copy, status=", (int)status);
}

void copy_spiky_to_cpu(const at::Tensor& src, at::Tensor& dst) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "Source must be nkipy");
  TORCH_CHECK(dst.is_cpu(), "Destination must be CPU");
  TORCH_CHECK(src.is_contiguous(), "Source nkipy tensor must be contiguous for copy to CPU");
  TORCH_CHECK(dst.is_contiguous(), "Destination CPU tensor must be contiguous");

  nrt_tensor_t* nrt_src = get_nrt_tensor(src);
  TORCH_CHECK(nrt_src != nullptr, "Failed to get nkipy tensor handle");
  size_t byte_offset = static_cast<size_t>(src.storage_offset()) * src.element_size();
  if (dst.nbytes() == 0) return;
  NRT_STATUS status = nrt_tensor_read(nrt_src, dst.data_ptr(), byte_offset, dst.nbytes());
  TORCH_CHECK(status == NRT_SUCCESS, "Failed nkipy->CPU copy, status=", (int)status);
}

void copy_spiky_to_spiky(const at::Tensor& src, at::Tensor& dst) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "Source must be nkipy");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1, "Destination must be nkipy");
  TORCH_CHECK(src.device().index() == dst.device().index(), "Cross-core copy not supported");
  TORCH_CHECK(src.is_contiguous(), "Source nkipy tensor must be contiguous");
  TORCH_CHECK(dst.is_contiguous(), "Destination nkipy tensor must be contiguous");

  nrt_tensor_t* nrt_src = get_nrt_tensor(src);
  nrt_tensor_t* nrt_dst = get_nrt_tensor(dst);
  TORCH_CHECK(nrt_src && nrt_dst, "Failed to get nkipy tensor handles");

  size_t src_off = static_cast<size_t>(src.storage_offset()) * src.element_size();
  size_t dst_off = static_cast<size_t>(dst.storage_offset()) * dst.element_size();
  TORCH_CHECK(src.nbytes() == dst.nbytes(), "Source and destination must have same bytes");
  if (src.nbytes() == 0) return;
  NRT_STATUS status = nrt_tensor_copy(nrt_src, src_off, nrt_dst, dst_off, src.nbytes());
  TORCH_CHECK(status == NRT_SUCCESS, "Failed nkipy->nkipy copy, status=", (int)status);
}

}  // namespace

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool /*non_blocking*/) {
  if (self.is_same(src)) return self;
  TORCH_CHECK(self.sizes() == src.sizes(), "copy_: sizes must match");
  bool same_type = self.dtype() == src.dtype();

  if (src.is_cpu() && self.device().type() == c10::DeviceType::PrivateUse1) {
    if (!same_type) {
      at::Tensor src_conv = src.to(self.dtype());
      copy_cpu_to_spiky(src_conv, self);
    } else {
      copy_cpu_to_spiky(src, self);
    }
    return self;
  }

  if (src.device().type() == c10::DeviceType::PrivateUse1 && self.is_cpu()) {
    if (!same_type) {
      at::Tensor tmp = at::empty(src.sizes(), self.options().dtype(src.dtype()));
      copy_spiky_to_cpu(src, tmp);
      self.copy_(tmp);
    } else {
      copy_spiky_to_cpu(src, self);
    }
    return self;
  }

  if (src.device().type() == c10::DeviceType::PrivateUse1 &&
      self.device().type() == c10::DeviceType::PrivateUse1) {
    if (!same_type) {
      at::Tensor tmp = src.to(at::kCPU, src.dtype());
      tmp = tmp.to(self.dtype());
      copy_cpu_to_spiky(tmp, self);
    } else {
      copy_spiky_to_spiky(src, self);
    }
    return self;
  }

  TORCH_CHECK(false, "Unsupported device combination for copy_");
}

at::Tensor _to_copy(const at::Tensor& self,
                    c10::optional<at::ScalarType> dtype,
                    c10::optional<c10::Layout> layout,
                    c10::optional<c10::Device> device,
                    c10::optional<bool> pin_memory, bool non_blocking,
                    c10::optional<c10::MemoryFormat> memory_format) {
  bool is_to_spiky = device.has_value() && device->type() == c10::DeviceType::PrivateUse1;
  bool is_from_spiky = self.device().type() == c10::DeviceType::PrivateUse1;

  if ((is_to_spiky || is_from_spiky) && memory_format.has_value() &&
      *memory_format != c10::MemoryFormat::Contiguous &&
      *memory_format != c10::MemoryFormat::Preserve) {
    TORCH_CHECK(false, "Spiky tensors only support contiguous memory format");
  }

  auto options = self.options();
  if (dtype) options = options.dtype(*dtype);
  if (layout) options = options.layout(*layout);
  if (device) options = options.device(*device);
  if (pin_memory) options = options.pinned_memory(*pin_memory);

  bool needs_copy = false;
  if (dtype && *dtype != self.dtype()) needs_copy = true;
  if (device && *device != self.device()) needs_copy = true;
  if (layout && *layout != self.layout()) needs_copy = true;

  if (!needs_copy && (!memory_format || *memory_format == c10::MemoryFormat::Preserve)) {
    return self;
  }

  if (is_to_spiky && !self.is_contiguous()) {
    at::Tensor cpu_contig = self.contiguous();
    at::Tensor result = at::empty(cpu_contig.sizes(), options);
    result.copy_(cpu_contig, non_blocking);
    return result;
  }

  at::Tensor result = at::empty(self.sizes(), options);
  result.copy_(self, non_blocking);
  return result;
}

at::Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(self.is_cpu() && dst.device().type() == c10::DeviceType::PrivateUse1,
              "_copy_from_and_resize only supports CPU->nkipy");
  if (dst.numel() == 0) {
    const_cast<at::Tensor&>(dst).resize_as_(self);
  }
  TORCH_CHECK(self.sizes() == dst.sizes(), "_copy_from_and_resize requires same sizes");
  const_cast<at::Tensor&>(dst).copy_(self);
  return dst;
}

static at::Tensor alias_with_sizes_and_strides(const at::Tensor& self,
                                               at::IntArrayRef sizes,
                                               at::IntArrayRef strides) {
  at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  auto* impl = result.unsafeGetTensorImpl();
  impl->set_storage_offset(self.storage_offset());
  impl->set_sizes_and_strides(sizes, strides);
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(stride.has_value(), "view size is not compatible");
  return alias_with_sizes_and_strides(self, inferred_size, *stride);
}

at::Tensor as_strided(const at::Tensor& self, at::IntArrayRef size,
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

at::Tensor _reshape_alias(const at::Tensor& self, at::IntArrayRef sizes,
                          at::IntArrayRef /*strides*/) {
  return view(self, sizes);
}

at::Tensor& fill_scalar(at::Tensor& self, const at::Scalar& value) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "fill_: expected nkipy tensor");
  at::Tensor cpu_tensor = at::empty_like(self, self.options().device(at::kCPU));
  cpu_tensor.fill_(value);
  self.copy_(cpu_tensor);
  return self;
}

}  // namespace ops
}  // namespace spiky_torch
