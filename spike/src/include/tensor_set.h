#ifndef SPIKE_SRC_INCLUDE_TENSOR_SET_H
#define SPIKE_SRC_INCLUDE_TENSOR_SET_H

#include "tensor.h"
#include <string>

namespace spike {

// RAII wrapper for NRT tensor set
class NrtTensorSet {
public:
  NrtTensorSet();
  ~NrtTensorSet();

  // Non-copyable, movable
  NrtTensorSet(const NrtTensorSet &) = delete;
  NrtTensorSet &operator=(const NrtTensorSet &) = delete;
  NrtTensorSet(NrtTensorSet &&other) noexcept;
  NrtTensorSet &operator=(NrtTensorSet &&other) noexcept;

  void add_tensor(const std::string &name, const NrtTensor &tensor);
  nrt_tensor_set_t *get_ptr() const { return ptr_; }

private:
  nrt_tensor_set_t *ptr_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_TENSOR_SET_H
