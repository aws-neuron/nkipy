#ifndef SPIKE_SRC_INCLUDE_SPIKE_H
#define SPIKE_SRC_INCLUDE_SPIKE_H

#include "model.h"
#include "nrt_wrapper.h"
#include "tensor.h"
#include "tensor_set.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nb = nanobind;

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

// NonBlock command structures
struct NonBlockCloseCmd {};

struct NonBlockTensorReadCmd {
  uint64_t id;
  std::shared_ptr<const NrtTensor> tensor;
  size_t offset;
  size_t size;
  void *data;
  std::variant<nb::bytes, nb::ndarray<>> data_obj;
};

struct NonBlockTensorWriteCmd {
  uint64_t id;
  std::shared_ptr<NrtTensor> tensor;
  const void *data;
  size_t size;
  std::variant<nb::bytes, nb::ndarray<>> data_obj;
  size_t offset;
};

struct NonBlockTensorWriteBatchedCmd {
  uint64_t id;
  uint64_t batch_id;
};

struct NonBlockTensorReadBatchedCmd {
  uint64_t id;
  uint64_t batch_id;
};

typedef std::variant<NonBlockTensorReadCmd, NonBlockTensorWriteCmd,
                     NonBlockTensorWriteBatchedCmd,
                     NonBlockTensorReadBatchedCmd, NonBlockCloseCmd>
    NonBlockTensorCmd;

struct NonBlockExecCmd {
  uint64_t id;
  std::shared_ptr<NrtModel> model;
  std::shared_ptr<const NrtTensorSet> input_set;
  std::shared_ptr<NrtTensorSet> output_set;
  std::optional<std::string> ntff_name;
  bool save_trace;
};

typedef std::variant<NonBlockExecCmd, NonBlockCloseCmd>
    NonBlockExecOrCloseCmd;

// Thread-safe queue template (blocking and non-blocking versions)
template <typename T, bool is_blocking> class LockedQueue {
public:
  LockedQueue() {
    if constexpr (!is_blocking) {
      size_ = 0;
    }
  }

  void push(const T &value) {
    std::unique_lock lk(mtx_);
    q_.push(value);
    lk.unlock();
    if constexpr (is_blocking) {
      cv_.notify_one();
    } else {
      size_.fetch_add(1, std::memory_order_release);
    }
  }

  void push(T &&value) {
    std::unique_lock lk(mtx_);
    q_.push(std::move(value));
    lk.unlock();
    if constexpr (is_blocking) {
      cv_.notify_one();
    } else {
      size_.fetch_add(1, std::memory_order_release);
    }
  }

  template <typename... Args> void emplace(Args &&...args) {
    std::unique_lock lk(mtx_);
    q_.emplace(std::forward<Args>(args)...);
    lk.unlock();
    if constexpr (is_blocking) {
      cv_.notify_one();
    } else {
      size_.fetch_add(1, std::memory_order_release);
    }
  }

  template <typename T_ = T> std::enable_if_t<is_blocking, T_> pop() {
    std::unique_lock lk(mtx_);
    cv_.wait(lk, [this]() { return !q_.empty(); });
    T ret(std::move(q_.front()));
    q_.pop();
    lk.unlock();
    return ret;
  }

  // This function is only safe with one consumer
  template <typename T_ = T>
  std::enable_if_t<!is_blocking, std::optional<T_>> try_pop() {
    if (size_.load(std::memory_order_acquire) >= 1) {
      size_.fetch_sub(1, std::memory_order_release);
      std::lock_guard lk(mtx_);
      T ret(std::move(q_.front()));
      q_.pop();
      return ret;
    } else {
      return std::nullopt;
    }
  }

private:
  struct Empty {};

  std::queue<T> q_;
  std::mutex mtx_;
  std::conditional_t<is_blocking, std::condition_variable, Empty> cv_;
  std::conditional_t<is_blocking, Empty, std::atomic_int64_t> size_;
};

// NonBlock result structures (exposed to Python)
struct NonBlockTensorReadResult {
  uint64_t id;
  std::variant<nb::bytes, nb::ndarray<>> data;
  std::optional<std::variant<SpikeError, NrtError>> err;
};

struct NonBlockTensorWriteResult {
  uint64_t id;
  std::optional<std::variant<SpikeError, NrtError>> err;
};

struct NonBlockExecResult {
  uint64_t id;
  std::optional<std::variant<SpikeError, NrtError>> err;
};

typedef std::variant<NonBlockTensorReadResult, NonBlockTensorWriteResult,
                     NonBlockExecResult>
    NonBlockResult;

// NonBlock internal result structures (used in worker threads to avoid GIL)
// These hold shared_ptrs to keep resources alive until convert_internal_result()
// transfers nanobind objects to Result types. This two-phase pattern ensures
// nanobind/Python object destructors run in GIL context, not in worker threads.
struct NonBlockTensorReadInternalResult {
  uint64_t id;
  std::variant<nb::bytes, nb::ndarray<>> data_obj;
  std::optional<std::variant<SpikeError, NrtError>> err;

  std::shared_ptr<const NrtTensor> tensor;
};

struct NonBlockTensorWriteInternalResult {
  uint64_t id;
  std::optional<std::variant<SpikeError, NrtError>> err;

  std::shared_ptr<NrtTensor> tensor;
  std::variant<nb::bytes, nb::ndarray<>> data_obj;
};

struct NonBlockExecInternalResult {
  uint64_t id;
  std::optional<std::variant<SpikeError, NrtError>> err;

  std::shared_ptr<const NrtModel> model;
  std::shared_ptr<const NrtTensorSet> input_set;
  std::shared_ptr<NrtTensorSet> output_set;
};

typedef std::variant<NonBlockTensorReadInternalResult,
                     NonBlockTensorWriteInternalResult,
                     NonBlockExecInternalResult>
    NonBlockInternalResult;

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

  // Nonblocking operations
  void init_nonblock();

  uint64_t tensor_write_nonblock(std::shared_ptr<NrtTensor> tensor,
                                  nb::bytes data_obj, size_t offset = 0);
  uint64_t tensor_write_nonblock(std::shared_ptr<NrtTensor> tensor,
                                  nb::ndarray<> data_obj,
                                  size_t offset = 0);
  uint64_t tensor_write_nonblock(std::shared_ptr<NrtTensor> tensor,
                                  const void *data, size_t size,
                                  size_t offset);

  uint64_t tensor_read_nonblock(std::shared_ptr<const NrtTensor> tensor,
                                 size_t offset = 0, size_t size = 0);
  uint64_t tensor_read_nonblock(std::shared_ptr<const NrtTensor> tensor,
                                 nb::ndarray<> dest, size_t offset = 0,
                                 size_t size = 0);

  uint64_t tensor_write_nonblock_batched_prepare(
      std::vector<std::shared_ptr<NrtTensor>> tensors,
      std::vector<nb::ndarray<>> data_objs,
      std::optional<std::vector<size_t>> offsets);
  uint64_t tensor_write_nonblock_batched_start(uint64_t batch_id);

  uint64_t tensor_read_nonblock_batched_prepare(
      std::vector<std::shared_ptr<const NrtTensor>> tensors,
      std::vector<nb::ndarray<>> dests,
      std::optional<std::vector<size_t>> offsets,
      std::optional<std::vector<size_t>> sizes);
  uint64_t tensor_read_nonblock_batched_start(uint64_t batch_id);

  uint64_t
  execute_nonblock(std::shared_ptr<NrtModel> model,
                   std::shared_ptr<const NrtTensorSet> input_set,
                   std::shared_ptr<NrtTensorSet> output_set,
                   std::optional<std::string> ntff_name = std::nullopt,
                   bool save_trace = false);

  std::optional<NonBlockResult> try_poll();

  NrtTensorSet create_tensor_set(
      const std::unordered_map<std::string, std::shared_ptr<const NrtTensor>>
          &tensor_map);

  // Wrap existing NRT objects (for interop with external code)
  NrtModel wrap_model(nrt_model_t *ptr);
  NrtTensor wrap_tensor(nrt_tensor_t *ptr);
  NrtTensorSet wrap_tensor_set(nrt_tensor_set_t *ptr);

  // Model introspection
  ModelTensorInfo get_tensor_info(NrtModel &model);

private:
  int verbose_level_;
  std::unique_ptr<NrtRuntime> runtime_;

  // Nonblock support
  uint64_t next_non_block_id_ = 0;
  uint64_t next_batch_id_ = 0;

  std::vector<std::thread> tensor_threads_;
  std::vector<std::thread> exec_threads_;
  std::vector<LockedQueue<NonBlockTensorCmd, true>> tensor_queues_;
  std::vector<LockedQueue<NonBlockExecOrCloseCmd, true>> exec_queues_;
  LockedQueue<NonBlockInternalResult, false> noti_queue_;

  std::unordered_map<uint64_t, std::vector<NonBlockTensorWriteCmd>>
      tensor_write_batched_cmds_;
  std::shared_mutex tensor_write_batched_cmds_mtx_;

  std::unordered_map<uint64_t, std::vector<NonBlockTensorReadCmd>>
      tensor_read_batched_cmds_;
  std::shared_mutex tensor_read_batched_cmds_mtx_;

  void loop_execute(int core_id);
  void loop_tensor(int core_id);

  // Helper methods
  NrtTensorSet create_tensor_set(
      const std::unordered_map<std::string, NrtTensor &> &tensor_map);
  std::string dtype_to_string(nrt_dtype_t dtype);
  NonBlockResult convert_internal_result(NonBlockInternalResult &internal_res_);
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_SPIKE_H
