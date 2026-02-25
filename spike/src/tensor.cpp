#include "tensor.h"
#include "nrt_wrapper.h"
#include "spike.h"
#include <sstream>

namespace spike {

// NrtTensor implementation
NrtTensor::NrtTensor(nrt_tensor_placement_t placement, uint32_t core_id,
                     size_t size, const std::string &name, const Spike *spike)
    : ptr_(nullptr), core_id_(core_id), size_(size), name_(name),
      spike_(spike) {

  NRT_STATUS status =
      nrt_tensor_allocate(placement, core_id, size, name.c_str(), &ptr_);
  if (status != 0) {
    ptr_ = nullptr;
    throw NrtError(status, "Failed to allocate tensor");
  }
}

NrtTensor::NrtTensor(const NrtTensor &source, size_t offset, size_t size,
                     const std::string &name)
    : ptr_(nullptr), core_id_(source.core_id_), size_(size), name_(name),
      spike_(source.spike_) {
  if (source.is_freed()) {
    throw SpikeError(
        "Unable to allocate tensor slice from a source tensor that is freed. "
        "Please check the lifetime of the source tensor.");
  }

  NRT_STATUS status = nrt_tensor_allocate_slice(source.get_ptr(), offset, size,
                                                name.c_str(), &ptr_);
  if (status != 0) {
    ptr_ = nullptr;
    throw NrtError(status, "Failed to allocate tensor slice");
  }
}

NrtTensor::~NrtTensor() { free(); }

NrtTensor::NrtTensor(NrtTensor &&other) noexcept
    : ptr_(other.ptr_), core_id_(other.core_id_), size_(other.size_),
      name_(std::move(other.name_)), spike_(other.spike_) {
  other.ptr_ = nullptr;
}

NrtTensor &NrtTensor::operator=(NrtTensor &&other) noexcept {
  if (this != &other) {
    free();
    ptr_ = other.ptr_;
    core_id_ = other.core_id_;
    name_ = std::move(other.name_);
    size_ = other.size_;
    spike_ = other.spike_;
    other.ptr_ = nullptr;
  }
  return *this;
}

bool NrtTensor::is_freed() const {
  return ptr_ == nullptr || (is_owner() && spike_->is_closed());
}

void NrtTensor::write(const void *data, size_t size, size_t offset) {
  if (is_freed()) {
    throw SpikeError("Unable to write to a freed tensor. Please check the "
                     "lifetime of the tensor.");
  }

  if (get_size() < (offset + size)) {
    std::stringstream ss;
    ss << "The write operation exceeds the tensor bound. "
       << "Tensor size: " << get_size() << " bytes; "
       << "Input size: " << size << " bytes; "
       << "Offset: " << offset << " bytes.";
    throw SpikeError(ss.str());
  }

  NRT_STATUS status = nrt_tensor_write(ptr_, data, offset, size);
  if (status != 0) {
    throw NrtError(status, "Failed to write tensor");
  }
}

void NrtTensor::read(void *data, size_t size, size_t offset) const {
  if (is_freed()) {
    throw SpikeError("Unable to read from a freed tensor. Please check the "
                     "lifetime of the tensor.");
  }

  if (get_size() < (offset + size)) {
    std::stringstream ss;
    ss << "The read operation exceeds the tensor bound. "
       << "Tensor size: " << get_size() << " bytes; "
       << "Read size: " << size << " bytes; "
       << "Offset: " << offset << " bytes.";
    throw SpikeError(ss.str());
  }

  NRT_STATUS status = nrt_tensor_read(ptr_, data, offset, size);
  if (status != 0) {
    throw NrtError(status, "Failed to read tensor");
  }
}

void NrtTensor::free() {
  if (is_freed() || !is_owner()) {
    // Do not throw exception as this is perfectly fine
    return;
  }

  nrt_tensor_free(&ptr_);
  ptr_ = nullptr;
}

std::string NrtTensor::to_string() const {
  std::ostringstream oss;
  oss << "NrtTensor(ptr=" << ptr_ << ", name='" << name_ << "', size=" << size_
      << ", core_id=" << core_id_
      << ", is_owner=" << (is_owner() ? "True" : "False") << ")";
  return oss.str();
}

} // namespace spike
