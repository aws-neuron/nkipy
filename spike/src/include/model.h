#ifndef SPIKE_SRC_INCLUDE_MODEL_H
#define SPIKE_SRC_INCLUDE_MODEL_H

#include "tensor_set.h"
#include <atomic>
#include <memory>
#include <optional>
#include <unordered_map>
#include <string>

namespace spike {

class Spike;

// Handle for an in-flight asynchronous model execution.
//
// Owns the input/output tensor sets and the completion-status slot for one
// scheduled execution, keeping them alive until the request completes. NRT
// executes sequences on the COMPUTE XU queue in submission order, so a caller
// that schedules a chain of executions only needs to wait() on the last handle
// before reading results back — but every handle in the chain must be kept
// alive until that wait returns, since NRT references the tensor sets while the
// work is in flight.
class AsyncExecution {
public:
  AsyncExecution(NrtTensorSet &&inputs, NrtTensorSet &&outputs,
                 nrta_seq_t sequence, std::unique_ptr<NRT_STATUS> ret);
  ~AsyncExecution() = default;

  // Non-copyable, movable
  AsyncExecution(const AsyncExecution &) = delete;
  AsyncExecution &operator=(const AsyncExecution &) = delete;
  AsyncExecution(AsyncExecution &&) = default;
  AsyncExecution &operator=(AsyncExecution &&) = default;

  nrta_seq_t sequence() const { return sequence_; }
  bool is_completed() const;
  // Blocks until the scheduled request completes, then throws NrtError if the
  // execution reported a failure. Releasing the GIL is the caller's job.
  void wait();

private:
  NrtTensorSet input_set_;
  NrtTensorSet output_set_;
  nrta_seq_t sequence_;
  std::unique_ptr<NRT_STATUS> ret_;
  bool waited_;
};

// RAII wrapper for NRT model
class NrtModel {
public:
  // This constructor creates an NrtModel that owns the model
  NrtModel(const std::string &neff_path, uint32_t core_id, bool cc_enabled,
           uint32_t rank_id, uint32_t world_size, const Spike *spike);
  // This constructor creates an NrtModel that references an existing model
  NrtModel(nrt_model_t *ptr, uint32_t core_id, bool cc_enabled,
           uint32_t rank_id, uint32_t world_size);
  ~NrtModel();

  // Non-copyable, movable
  NrtModel(const NrtModel &) = delete;
  NrtModel &operator=(const NrtModel &) = delete;
  NrtModel(NrtModel &&other) noexcept;
  NrtModel &operator=(NrtModel &&other) noexcept;

  // Getters (read-only properties for Python)
  nrt_model_t *get_ptr() const { return ptr_; }
  const std::string &get_neff_path() const { return neff_path_; }
  uint32_t get_core_id() const { return core_id_; }
  uint32_t get_rank_id() const { return rank_id_; }
  uint32_t get_world_size() const { return world_size_; }
  bool get_is_collective() const { return is_collective_; }
  bool is_unloaded() const;
  bool is_owner() const { return runtime_closed_ != nullptr; }

  // String representation
  std::string to_string() const;

  void execute(const NrtTensorSet &inputs, NrtTensorSet &outputs,
               std::optional<std::string> ntff_name, bool save_trace);
  // Schedules an asynchronous model execution. Takes ownership of the input and
  // output tensor sets and returns an AsyncExecution that keeps them alive
  // until the request completes.
  AsyncExecution execute_schedule(NrtTensorSet &&inputs, NrtTensorSet &&outputs);
  nrt_tensor_info_array_t *get_tensor_info();
  static void free_tensor_info(nrt_tensor_info_array_t *info);
  void unload();

private:
  nrt_model_t *ptr_;
  std::string neff_path_;
  uint32_t core_id_;
  uint32_t rank_id_;
  uint32_t world_size_;
  bool is_collective_;
  std::shared_ptr<std::atomic_bool> runtime_closed_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_MODEL_H
