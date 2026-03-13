#ifndef SPIKE_SRC_INCLUDE_TENSOR_SET_H
#define SPIKE_SRC_INCLUDE_TENSOR_SET_H

#include "tensor.h"
#include <memory>
#include <string>
#include <vector>

namespace spike {

class Spike;

// RAII wrapper for NRT tensor set
class NrtTensorSet {
public:
  // Constructor that creates an owned tensor set
  explicit NrtTensorSet(const Spike *spike);
  // Constructor to wrap an existing tensor set (non-owning)
  explicit NrtTensorSet(nrt_tensor_set_t *ptr);
  ~NrtTensorSet();

  // Non-copyable, movable
  NrtTensorSet(const NrtTensorSet &) = delete;
  NrtTensorSet &operator=(const NrtTensorSet &) = delete;
  NrtTensorSet(NrtTensorSet &&other) noexcept;
  NrtTensorSet &operator=(NrtTensorSet &&other) noexcept;

  bool is_freed() const;
  bool is_owner() const { return spike_ != nullptr; }

  void add_tensor(const std::string &name,
                  std::shared_ptr<const NrtTensor> tensor);
  void add_tensor(const std::string &name, const NrtTensor &tensor);
  nrt_tensor_set_t *get_ptr() const { return ptr_; }
  void free();

private:
  nrt_tensor_set_t *ptr_;
  const Spike *spike_;
  std::vector<std::shared_ptr<const NrtTensor>> tensors_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_TENSOR_SET_H
