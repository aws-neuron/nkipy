#ifndef SPIKE_SRC_INCLUDE_SPIKE_H
#define SPIKE_SRC_INCLUDE_SPIKE_H

#include "model.h"
#include "nrt_wrapper.h"
#include "tensor.h"
#include "tensor_set.h"

#include <nrt/nrt_async.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
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

// Prepared-batch records. These only hold the arguments to be used when
// `_batched_start` is called; at start time we just submit one nrta_* request
// per entry and group them under a single cmd_id.
struct PreparedTensorWrite {
  std::shared_ptr<NrtTensor> tensor;
  const void *data;
  size_t size;
  size_t offset;
  std::variant<nb::bytes, nb::ndarray<>> data_obj;
};

struct PreparedTensorRead {
  std::shared_ptr<const NrtTensor> tensor;
  size_t offset;
  size_t size;
  void *data;
  std::variant<nb::bytes, nb::ndarray<>> data_obj;
};

// Pending nonblocking operations. Each cmd_id we hand back to Python maps to
// one PendingOp. The id and wait_seq are common to every kind of op, so they
// live directly on PendingOp; the per-kind payload is the variant below. The
// batched variants hold N sub-requests submitted back-to-back on the same
// (lnc, xu, queue=0); their seq numbers are consecutive, so wait_seq stores
// the last one and a completed wait_seq implies every prior sub-request is
// complete too.

// The nrta_* APIs store the op's completion status through the `ret` pointer
// *after* submission, so `ret` must outlive the call. Callers therefore enqueue
// the PendingOp first and pass a pointer into the deque-resident copy (see
// enqueue_pending); std::deque never relocates existing elements on push_back,
// so that address stays valid until the op is harvested.
struct PendingTensorWrite {
  NRT_STATUS ret;
  std::shared_ptr<NrtTensor> tensor;
  // Anchors the Python-owned source buffer until the op completes. Empty for
  // the raw-pointer overload (caller manages the buffer lifetime).
  std::optional<std::variant<nb::bytes, nb::ndarray<>>> data_obj;
};

struct PendingTensorRead {
  NRT_STATUS ret;
  std::shared_ptr<const NrtTensor> tensor;
  // The destination buffer; also returned to Python via
  // NonBlockTensorReadResult.data.
  std::variant<nb::bytes, nb::ndarray<>> data_obj;
};

struct PendingTensorWriteBatched {
  // Lifetime anchors live in tensor_write_batched_prepared_[batch_id], which
  // persists until close() (or a future explicit-release API) so the same
  // prepared batch can be _start'd many times.
  uint64_t batch_id;
  std::vector<NRT_STATUS> rets;
};

struct PendingTensorReadBatched {
  uint64_t batch_id;
  std::vector<NRT_STATUS> rets;
};

struct PendingExecute {
  NRT_STATUS ret;
  std::shared_ptr<NrtModel> model;
  std::shared_ptr<const NrtTensorSet> input_set;
  std::shared_ptr<NrtTensorSet> output_set;
};

struct PendingOp {
  uint64_t id;
  nrta_seq_t wait_seq;
  std::variant<PendingTensorWrite, PendingTensorRead,
               PendingTensorWriteBatched, PendingTensorReadBatched,
               PendingExecute>
      op;
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

  // Nonblock state
  uint64_t next_non_block_id_ = 0;
  uint64_t next_batch_id_ = 0;

  // One pending-op queue per (lnc, xu) channel. Each queue is FIFO by
  // submission order, and within a queue nrta_seq_t values are monotonically
  // increasing, so a channel only needs its front op's wait_seq checked
  // against nrta_get_sequence's latest-completed seq for that channel.
  static constexpr uint32_t MAX_LNC = 128;
  static constexpr uint32_t NUM_CHANNELS = MAX_LNC * NRTA_XU_TYPE_NUM;
  std::array<std::array<std::deque<PendingOp>, NRTA_XU_TYPE_NUM>, MAX_LNC>
      xu_queues_;

  // epoll-based completion multiplexing. Each (lnc, xu) channel that has ever
  // had work gets one eventfd, registered with the runtime via
  // nrta_event_register_xu_completion() and added to a single epoll instance.
  // The runtime signals a channel's eventfd whenever that XU completes any
  // sequence, so try_poll() can ask epoll which channels made progress instead
  // of probing all MAX_LNC * NRTA_XU_TYPE_NUM channels every call.
  int epoll_fd_ = -1;
  // Per-channel eventfd, -1 until the channel is first registered.
  std::array<int, NUM_CHANNELS> channel_event_fds_;

  // Channels whose front op may be ready to harvest. Populated by epoll_wait
  // (each signaled eventfd) and by try_poll itself (a channel stays queued
  // after a successful harvest, since one eventfd signal can cover several
  // completed ops). Removed once the front is found not-yet-complete or the
  // queue empties.
  //
  // A FIFO queue (rather than an ordered set) gives round-robin fairness: every
  // try_poll() pops the front channel, and a channel that still has pollable
  // work is pushed to the back, so a continuously-busy low-index channel can't
  // starve the others. scan_channel_queued_ tracks membership in O(1) so a
  // channel is never enqueued twice.
  std::deque<uint32_t> scan_channels_;
  std::array<bool, NUM_CHANNELS> scan_channel_queued_;

  static constexpr uint32_t channel_index(uint32_t lnc, uint32_t xu) {
    return lnc * NRTA_XU_TYPE_NUM + xu;
  }

  // Lazily creates the epoll instance and the channel's eventfd, registering
  // the latter with the runtime and the epoll set. Idempotent per channel.
  void ensure_channel_registered(uint32_t lnc, nrta_xu_t xu);

  // Prepared batches (data kept alive between _prepare and _start).
  std::unordered_map<uint64_t, std::vector<PreparedTensorWrite>>
      tensor_write_batched_prepared_;

  std::unordered_map<uint64_t, std::vector<PreparedTensorRead>>
      tensor_read_batched_prepared_;

  // Appends op to the (lnc, xu) queue and returns a reference to the enqueued
  // element. std::deque never relocates existing elements on push_back, so the
  // returned reference (and pointers into its ret field) stay valid until the
  // op is popped in try_poll. Scalar callers enqueue *before* submitting so the
  // nrta_* call can write completion status straight into the queued op.
  PendingOp &enqueue_pending(uint32_t lnc, nrta_xu_t xu, PendingOp op);

  // Helper methods
  NrtTensorSet create_tensor_set(
      const std::unordered_map<std::string, NrtTensor &> &tensor_map);
  std::string dtype_to_string(nrt_dtype_t dtype);

  static std::optional<std::variant<SpikeError, NrtError>>
  ret_to_err(NRT_STATUS ret);
  static std::optional<std::variant<SpikeError, NrtError>>
  rets_to_err(const std::vector<NRT_STATUS> &rets);
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_SPIKE_H
