#include "tensor_set.h"

namespace spike {

NrtTensorSet::NrtTensorSet(std::shared_ptr<std::atomic_bool> runtime_closed)
    : ptr_(nullptr), runtime_closed_(std::move(runtime_closed)) {
  NRT_STATUS status = nrt_allocate_tensor_set(reinterpret_cast<void **>(&ptr_));
  if (status != 0) {
    throw NrtError(status, "Failed to allocate tensor set");
  }
}

void NrtTensorSet::destroy() noexcept {
  if (!ptr_) {
    return;
  }
  // Skip the NRT call if the runtime is already closed: cleanup happens
  // implicitly at nrt_close, and calling nrt_destroy_tensor_set afterwards
  // logs an "Unexpected runtime state" error.
  if (!runtime_closed_ || !runtime_closed_->load()) {
    nrt_destroy_tensor_set(reinterpret_cast<void **>(&ptr_));
  }
  ptr_ = nullptr;
}

NrtTensorSet::~NrtTensorSet() { destroy(); }

NrtTensorSet::NrtTensorSet(NrtTensorSet &&other) noexcept
    : ptr_(other.ptr_), runtime_closed_(std::move(other.runtime_closed_)) {
  other.ptr_ = nullptr;
}

NrtTensorSet &NrtTensorSet::operator=(NrtTensorSet &&other) noexcept {
  if (this != &other) {
    destroy();
    ptr_ = other.ptr_;
    runtime_closed_ = std::move(other.runtime_closed_);
    other.ptr_ = nullptr;
  }
  return *this;
}

void NrtTensorSet::add_tensor(const std::string &name,
                              const NrtTensor &tensor) {
  NRT_STATUS status =
      nrt_add_tensor_to_tensor_set(ptr_, name.c_str(), tensor.get_ptr());
  if (status != 0) {
    throw NrtError(status, "Failed to add tensor to set");
  }
}

} // namespace spike
