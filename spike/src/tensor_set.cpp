#include "tensor_set.h"
#include "spike.h"

namespace spike {

NrtTensorSet::NrtTensorSet(const Spike *spike)
    : ptr_(nullptr), spike_(spike) {
  NRT_STATUS status = nrt_allocate_tensor_set(reinterpret_cast<void **>(&ptr_));
  if (status != 0) {
    throw NrtError(status, "Failed to allocate tensor set");
  }
}

NrtTensorSet::NrtTensorSet(nrt_tensor_set_t *ptr)
    : ptr_(ptr), spike_(nullptr) {}

NrtTensorSet::~NrtTensorSet() { free(); }

NrtTensorSet::NrtTensorSet(NrtTensorSet &&other) noexcept
    : ptr_(other.ptr_), spike_(other.spike_),
      tensors_(std::move(other.tensors_)) {
  other.ptr_ = nullptr;
}

NrtTensorSet &NrtTensorSet::operator=(NrtTensorSet &&other) noexcept {
  if (this != &other) {
    free();
    ptr_ = other.ptr_;
    spike_ = other.spike_;
    tensors_ = std::move(other.tensors_);
    other.ptr_ = nullptr;
  }
  return *this;
}

bool NrtTensorSet::is_freed() const {
  return ptr_ == nullptr || (is_owner() && spike_->is_closed());
}

void NrtTensorSet::add_tensor(const std::string &name,
                               std::shared_ptr<const NrtTensor> tensor) {
  add_tensor(name, *tensor);
  tensors_.push_back(std::move(tensor));
}

void NrtTensorSet::add_tensor(const std::string &name,
                               const NrtTensor &tensor) {
  NRT_STATUS status =
      nrt_add_tensor_to_tensor_set(ptr_, name.c_str(), tensor.get_ptr());
  if (status != 0) {
    throw NrtError(status, "Failed to add tensor to set");
  }
}

void NrtTensorSet::free() {
  if (is_freed() || !is_owner()) {
    // Do not throw exception as this is perfectly fine
    return;
  }

  nrt_destroy_tensor_set(reinterpret_cast<void **>(&ptr_));
  ptr_ = nullptr;
}

} // namespace spike
