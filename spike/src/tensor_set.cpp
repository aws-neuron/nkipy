#include "tensor_set.h"

namespace spike {

NrtTensorSet::NrtTensorSet() : ptr_(nullptr) {
  NRT_STATUS status = nrt_allocate_tensor_set(reinterpret_cast<void **>(&ptr_));
  if (status != 0) {
    throw NrtError(status, "Failed to allocate tensor set");
  }
}

NrtTensorSet::~NrtTensorSet() {
  if (ptr_) {
    nrt_destroy_tensor_set(reinterpret_cast<void **>(&ptr_));
  }
}

NrtTensorSet::NrtTensorSet(NrtTensorSet &&other) noexcept : ptr_(other.ptr_) {
  other.ptr_ = nullptr;
}

NrtTensorSet &NrtTensorSet::operator=(NrtTensorSet &&other) noexcept {
  if (this != &other) {
    if (ptr_) {
      nrt_destroy_tensor_set(reinterpret_cast<void **>(&ptr_));
    }
    ptr_ = other.ptr_;
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
