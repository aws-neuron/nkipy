#ifndef SPIKE_SRC_INCLUDE_SPIKE_H
#define SPIKE_SRC_INCLUDE_SPIKE_H

#include "model.h"
#include "nrt_wrapper.h"
#include "tensor.h"
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace spike {

// Tensor metadata structure
struct TensorMetadata {
  size_t size;                // Tensor size in bytes
  std::string dtype;          // Data type as human-readable string
  std::vector<int64_t> shape; // Tensor shape dimensions
};

// Model tensor information structure
struct ModelTensorInfo {
  std::unordered_map<std::string, TensorMetadata> inputs;
  std::unordered_map<std::string, TensorMetadata> outputs;
};

// Main Spike class - Python interface
class Spike {
public:
  explicit Spike(int verbose_level = 0);
  ~Spike();

  // Non-copyable, movable
  Spike(const Spike &) = delete;
  Spike &operator=(const Spike &) = delete;
  Spike(Spike &&) = default;
  Spike &operator=(Spike &&) = default;

  // Static methods
  static uint32_t get_visible_neuron_core_count();

  // Runtime management
  int close();
  bool is_closed() const { return runtime_.get() == nullptr; }

  // Model operations
  NrtModel load_model(const std::string &neff_file, uint32_t core_id = 0,
                      bool cc_enabled = false, uint32_t rank_id = 0,
                      uint32_t world_size = 1);
  void unload_model(NrtModel &model);

  // Tensor operations
  NrtTensor allocate_tensor(size_t size, uint32_t core_id = 0,
                            std::optional<std::string> name = std::nullopt);
  NrtTensor slice_from_tensor(const NrtTensor &source, size_t offset = 0,
                              size_t size = 0,
                              std::optional<std::string> name = std::nullopt);
  void free_tensor(NrtTensor &tensor);

  // Tensor I/O operations
  void tensor_write(NrtTensor &tensor, const void *data, size_t data_size,
                    size_t offset = 0);
  std::vector<uint8_t> tensor_read(const NrtTensor &tensor, size_t offset = 0,
                                   size_t size = 0);
  void tensor_write_from_pybuffer(NrtTensor &tensor, const void *data,
                                  size_t data_size, size_t offset = 0);

  // Model execution
  void execute(NrtModel &model,
               const std::unordered_map<std::string, NrtTensor &> &inputs,
               const std::unordered_map<std::string, NrtTensor &> &outputs,
               std::optional<std::string> ntff_name = std::nullopt,
               bool save_trace = false);

  // Model introspection
  ModelTensorInfo get_tensor_info(NrtModel &model);

private:
  int verbose_level_;
  std::unique_ptr<NrtRuntime> runtime_;

  // Helper methods
  NrtTensorSet create_tensor_sets(
      const std::unordered_map<std::string, NrtTensor &> &tensor_map);
  std::string dtype_to_string(nrt_dtype_t dtype);
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_SPIKE_H
