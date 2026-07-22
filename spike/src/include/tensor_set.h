#ifndef SPIKE_SRC_INCLUDE_TENSOR_SET_H
#define SPIKE_SRC_INCLUDE_TENSOR_SET_H

#include "tensor.h"
#include <atomic>
#include <memory>
#include <string>

namespace spike {

// RAII wrapper for NRT tensor set
class NrtTensorSet {
public:
  // runtime_closed tracks Spike teardown so the destructor can skip
  // nrt_destroy_tensor_set once NRT is gone (an AsyncExecution may hold a set
  // past runtime close). Pass nullptr for a set whose lifetime is always within
  // an NRT call.
  explicit NrtTensorSet(
      std::shared_ptr<std::atomic_bool> runtime_closed = nullptr);
  ~NrtTensorSet();

  // Non-copyable, movable
  NrtTensorSet(const NrtTensorSet &) = delete;
  NrtTensorSet &operator=(const NrtTensorSet &) = delete;
  NrtTensorSet(NrtTensorSet &&other) noexcept;
  NrtTensorSet &operator=(NrtTensorSet &&other) noexcept;

  void add_tensor(const std::string &name, const NrtTensor &tensor);
  nrt_tensor_set_t *get_ptr() const { return ptr_; }

private:
  void destroy() noexcept;

  nrt_tensor_set_t *ptr_;
  std::shared_ptr<std::atomic_bool> runtime_closed_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_TENSOR_SET_H
