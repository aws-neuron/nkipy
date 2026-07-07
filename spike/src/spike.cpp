#include "spike.h"
#include <Python.h>
#include <cstring>
#include <iostream>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

namespace spike {

// Static guard to prevent multiple alive Spike instances
static bool g_alive_spike_instance_exists = false;

Spike::Spike(int verbose_level)
    : verbose_level_(verbose_level), runtime_(nullptr) {
  if (g_alive_spike_instance_exists) {
    throw SpikeError(
        "Cannot create Spike instance: a previous instance still exists. "
        "Call close() on the existing instance or delete spike with all "
        "tensors/models first.");
  }

  // Channel eventfds are created lazily on first use; -1 means unregistered.
  channel_event_fds_.fill(-1);
  scan_channel_queued_.fill(false);

  // RAII: Initialize NRT in constructor
  if (verbose_level_ > 0) {
    std::cout << "Initializing SPIKE Runtime" << std::endl;
  }

  runtime_ = std::make_unique<NrtRuntime>("spike", "1.0", true);
  g_alive_spike_instance_exists = true;
}

Spike::~Spike() {
  if (runtime_) {
    close();
  }
}

uint32_t Spike::get_visible_neuron_core_count() {
  return NrtRuntime::get_visible_nc_count();
}

int Spike::close() {
  for (auto &per_lnc : xu_queues_) {
    for (auto &q : per_lnc) {
      q.clear();
    }
  }
  tensor_write_batched_prepared_.clear();
  tensor_read_batched_prepared_.clear();

  // Tear down the epoll multiplexing state. Deregister each channel's eventfd
  // from the runtime (negative fd deregisters) before closing it.
  for (uint32_t lnc = 0; lnc < MAX_LNC; ++lnc) {
    for (uint32_t xu = 0; xu < NRTA_XU_TYPE_NUM; ++xu) {
      int &fd = channel_event_fds_[channel_index(lnc, xu)];
      if (fd < 0) {
        continue;
      }
      nrta_event_register_xu_completion(static_cast<int>(lnc),
                                        static_cast<nrta_xu_t>(xu),
                                        /*queue=*/0, /*fd=*/-1);
      ::close(fd);
      fd = -1;
    }
  }
  if (epoll_fd_ >= 0) {
    ::close(epoll_fd_);
    epoll_fd_ = -1;
  }
  scan_channels_.clear();
  scan_channel_queued_.fill(false);

  runtime_.reset();
  g_alive_spike_instance_exists = false;
  return 0;
}

void Spike::ensure_channel_registered(uint32_t lnc, nrta_xu_t xu) {
  if (lnc >= MAX_LNC) {
    throw SpikeError("LNC index exceeds MAX_LNC.");
  }
  if (xu >= NRTA_XU_TYPE_NUM) {
    throw SpikeError("XU index out of range.");
  }

  if (epoll_fd_ < 0) {
    epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd_ < 0) {
      throw SpikeError(std::string("epoll_create1 failed: ") +
                       std::strerror(errno));
    }
  }

  int &fd = channel_event_fds_[channel_index(lnc, xu)];
  if (fd >= 0) {
    return; // Already registered.
  }

  fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (fd < 0) {
    throw SpikeError(std::string("eventfd failed: ") + std::strerror(errno));
  }

  // The data carries the channel index so epoll_wait tells us exactly which
  // (lnc, xu) channel signaled.
  struct epoll_event ev;
  ev.events = EPOLLIN;
  ev.data.u32 = channel_index(lnc, xu);
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
    int err = errno;
    ::close(fd);
    fd = -1;
    throw SpikeError(std::string("epoll_ctl ADD failed: ") +
                     std::strerror(err));
  }

  // Hand the eventfd to the runtime; it signals it whenever this XU completes
  // any sequence on queue 0.
  NRT_STATUS status = nrta_event_register_xu_completion(
      static_cast<int>(lnc), xu, /*queue=*/0, fd);
  if (status != NRT_SUCCESS) {
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
    ::close(fd);
    fd = -1;
    throw NrtError(status, "Failed to register XU completion event");
  }
}

PendingOp &Spike::enqueue_pending(uint32_t lnc, nrta_xu_t xu, PendingOp op) {
  // Channel registration happens before submission (in the calling
  // tensor_*/execute_nonblock helpers), so we know an eventfd is in place for
  // every completion. xu must be the same XU the channel was registered on so
  // the queue index here matches the one try_poll derives from the epoll event.
  if (lnc >= MAX_LNC || xu >= NRTA_XU_TYPE_NUM) {
    throw SpikeError("Channel index out of range.");
  }
  std::deque<PendingOp> &q = xu_queues_[lnc][xu];
  q.push_back(std::move(op));
  return q.back();
}

NrtModel Spike::load_model(const std::string &neff_file, uint32_t core_id,
                           bool cc_enabled, uint32_t rank_id,
                           uint32_t world_size) {
  return NrtModel(neff_file, core_id, cc_enabled, rank_id, world_size, this);
}

void Spike::unload_model(NrtModel &model) { model.unload(); }

NrtTensor Spike::allocate_tensor(size_t size, uint32_t core_id,
                                 std::optional<std::string> name) {
  std::string actual_name = name.value_or("unnamed_tensor");
  return NrtTensor(NRT_TENSOR_PLACEMENT_DEVICE, core_id, size, actual_name,
                   this);
}

NrtTensor Spike::slice_from_tensor(const NrtTensor &source, size_t offset,
                                   size_t size,
                                   std::optional<std::string> name) {
  size_t actual_size = (size == 0) ? (source.get_size() - offset) : size;
  std::string actual_name = name.value_or("unnamed_tensor");

  return NrtTensor(source, offset, actual_size, actual_name);
}

void Spike::free_tensor(NrtTensor &tensor) { tensor.free(); }

void Spike::tensor_write(NrtTensor &tensor, const void *data, size_t data_size,
                         size_t offset) {
  tensor.write(data, data_size, offset);
}

std::vector<uint8_t> Spike::tensor_read(const NrtTensor &tensor, size_t offset,
                                        size_t size) {
  size_t actual_size = (size == 0) ? (tensor.get_size() - offset) : size;

  std::vector<uint8_t> result(actual_size);
  tensor.read(result.data(), actual_size, offset);
  return result;
}

void Spike::tensor_write_from_pybuffer(NrtTensor &tensor, const void *data,
                                       size_t data_size, size_t offset) {
  tensor_write(tensor, data, data_size, offset);
}

// Create tensor set with non-owning references (for synchronous operations).
// Caller must ensure tensors remain valid during use. Lighter weight alternative.
NrtTensorSet Spike::create_tensor_set(
    const std::unordered_map<std::string, NrtTensor &> &tensor_map) {
  NrtTensorSet tensor_set(this);

  for (const auto &[name, tensor] : tensor_map) {
    // FIXME: consider time-of-check-time-of-use (TOCTOU) race condition
    if (tensor.is_freed()) {
      throw SpikeError("Tensor '" + name +
                       "' is freed. Unable to add it into the tensor set. "
                       "Please check the lifetime of the tensor.");
    }
    tensor_set.add_tensor(name, tensor);
  }

  return tensor_set;
}

// Create tensor set with shared ownership (for async/nonblocking operations).
// Takes shared_ptrs to prevent premature tensor destruction during deferred
// execution in worker threads. Essential for correct async lifetime management.
NrtTensorSet Spike::create_tensor_set(
    const std::unordered_map<std::string, std::shared_ptr<const NrtTensor>>
        &tensor_map) {
  NrtTensorSet tensor_set(this);

  for (const auto &[name, tensor] : tensor_map) {
    if (tensor->is_freed()) {
      throw SpikeError("Tensor '" + name +
                       "' is freed. Unable to add it into the tensor set. "
                       "Please check the lifetime of the tensor.");
    }
    tensor_set.add_tensor(name, tensor);
  }

  return tensor_set;
}

void Spike::execute(NrtModel &model,
                    const std::unordered_map<std::string, NrtTensor &> &inputs,
                    const std::unordered_map<std::string, NrtTensor &> &outputs,
                    std::optional<std::string> ntff_name, bool save_trace) {
  // Create tensor sets
  NrtTensorSet input_set = create_tensor_set(inputs);
  NrtTensorSet output_set = create_tensor_set(outputs);

  // Execute
  model.execute(input_set, output_set, ntff_name, save_trace);
}

uint64_t Spike::tensor_write_nonblock(std::shared_ptr<NrtTensor> tensor,
                                       nb::bytes data_obj,
                                       size_t offset) {
  const void *data = data_obj.data();
  size_t size = data_obj.size();

  uint64_t cmd_id = next_non_block_id_++;
  uint32_t lnc = tensor->get_core_id();

  PendingTensorWrite p;
  p.ret = NRT_SUCCESS;
  p.tensor = tensor;
  p.data_obj = std::move(data_obj);

  ensure_channel_registered(lnc, NRTA_XU_TENSOR_OP);
  // Enqueue first so nrta_* can write completion status into the deque-resident
  // op (its address must outlive this call); roll back if scheduling fails.
  PendingOp &enq = enqueue_pending(
      lnc, NRTA_XU_TENSOR_OP, PendingOp{cmd_id, /*wait_seq=*/0, std::move(p)});
  PendingTensorWrite &q = std::get<PendingTensorWrite>(enq.op);
  NRT_STATUS status = nrta_tensor_write(
      tensor->get_ptr(), data, offset, size, static_cast<int>(lnc),
      /*queue=*/0, &q.ret, &enq.wait_seq);
  if (status != NRT_SUCCESS) {
    xu_queues_[lnc][NRTA_XU_TENSOR_OP].pop_back();
    throw NrtError(status, "Failed to schedule nonblocking tensor write");
  }

  return cmd_id;
}

uint64_t Spike::tensor_write_nonblock(std::shared_ptr<NrtTensor> tensor,
                                       nb::ndarray<> data_obj,
                                       size_t offset) {
  const void *data = data_obj.data();
  size_t size = data_obj.nbytes();

  uint64_t cmd_id = next_non_block_id_++;
  uint32_t lnc = tensor->get_core_id();

  PendingTensorWrite p;
  p.ret = NRT_SUCCESS;
  p.tensor = tensor;
  p.data_obj = std::move(data_obj);

  ensure_channel_registered(lnc, NRTA_XU_TENSOR_OP);
  PendingOp &enq = enqueue_pending(
      lnc, NRTA_XU_TENSOR_OP, PendingOp{cmd_id, /*wait_seq=*/0, std::move(p)});
  PendingTensorWrite &q = std::get<PendingTensorWrite>(enq.op);
  NRT_STATUS status = nrta_tensor_write(
      tensor->get_ptr(), data, offset, size, static_cast<int>(lnc),
      /*queue=*/0, &q.ret, &enq.wait_seq);
  if (status != NRT_SUCCESS) {
    xu_queues_[lnc][NRTA_XU_TENSOR_OP].pop_back();
    throw NrtError(status, "Failed to schedule nonblocking tensor write");
  }

  return cmd_id;
}

uint64_t Spike::tensor_write_nonblock(std::shared_ptr<NrtTensor> tensor,
                                       const void *data, size_t size,
                                       size_t offset) {
  uint64_t cmd_id = next_non_block_id_++;
  uint32_t lnc = tensor->get_core_id();

  PendingTensorWrite p;
  p.ret = NRT_SUCCESS;
  p.tensor = tensor;
  // No data_obj: caller manages the raw buffer's lifetime.

  ensure_channel_registered(lnc, NRTA_XU_TENSOR_OP);
  PendingOp &enq = enqueue_pending(
      lnc, NRTA_XU_TENSOR_OP, PendingOp{cmd_id, /*wait_seq=*/0, std::move(p)});
  PendingTensorWrite &q = std::get<PendingTensorWrite>(enq.op);
  NRT_STATUS status = nrta_tensor_write(
      tensor->get_ptr(), data, offset, size, static_cast<int>(lnc),
      /*queue=*/0, &q.ret, &enq.wait_seq);
  if (status != NRT_SUCCESS) {
    xu_queues_[lnc][NRTA_XU_TENSOR_OP].pop_back();
    throw NrtError(status, "Failed to schedule nonblocking tensor write");
  }

  return cmd_id;
}

uint64_t Spike::tensor_read_nonblock(std::shared_ptr<const NrtTensor> tensor,
                                      size_t offset, size_t size) {
  size = (size == 0) ? (tensor->get_size() - offset) : size;

  uint64_t cmd_id = next_non_block_id_++;
  uint32_t lnc = tensor->get_core_id();

  nb::bytes dest(nullptr, size);
  void *data = PyBytes_AsString(dest.ptr());

  PendingTensorRead p;
  p.ret = NRT_SUCCESS;
  p.tensor = tensor;
  p.data_obj = std::move(dest);

  ensure_channel_registered(lnc, NRTA_XU_TENSOR_OP);
  PendingOp &enq = enqueue_pending(
      lnc, NRTA_XU_TENSOR_OP, PendingOp{cmd_id, /*wait_seq=*/0, std::move(p)});
  PendingTensorRead &q = std::get<PendingTensorRead>(enq.op);
  NRT_STATUS status = nrta_tensor_read(
      data, tensor->get_ptr(), offset, size, static_cast<int>(lnc),
      /*queue=*/0, &q.ret, &enq.wait_seq);
  if (status != NRT_SUCCESS) {
    xu_queues_[lnc][NRTA_XU_TENSOR_OP].pop_back();
    throw NrtError(status, "Failed to schedule nonblocking tensor read");
  }

  return cmd_id;
}

uint64_t Spike::tensor_read_nonblock(std::shared_ptr<const NrtTensor> tensor,
                                      nb::ndarray<> dest, size_t offset,
                                      size_t size) {
  size = (size == 0) ? (tensor->get_size() - offset) : size;

  if (dest.nbytes() < size) {
    throw SpikeError("The read operation exceeds the destination bound.");
  }

  uint64_t cmd_id = next_non_block_id_++;
  uint32_t lnc = tensor->get_core_id();

  void *data = dest.data();

  PendingTensorRead p;
  p.ret = NRT_SUCCESS;
  p.tensor = tensor;
  p.data_obj = std::move(dest);

  ensure_channel_registered(lnc, NRTA_XU_TENSOR_OP);
  PendingOp &enq = enqueue_pending(
      lnc, NRTA_XU_TENSOR_OP, PendingOp{cmd_id, /*wait_seq=*/0, std::move(p)});
  PendingTensorRead &q = std::get<PendingTensorRead>(enq.op);
  NRT_STATUS status = nrta_tensor_read(
      data, tensor->get_ptr(), offset, size, static_cast<int>(lnc),
      /*queue=*/0, &q.ret, &enq.wait_seq);
  if (status != NRT_SUCCESS) {
    xu_queues_[lnc][NRTA_XU_TENSOR_OP].pop_back();
    throw NrtError(status, "Failed to schedule nonblocking tensor read");
  }

  return cmd_id;
}

uint64_t Spike::tensor_write_nonblock_batched_prepare(
    std::vector<std::shared_ptr<NrtTensor>> tensors,
    std::vector<nb::ndarray<>> data_objs,
    std::optional<std::vector<size_t>> offsets) {
  if (tensors.size() == 0) {
    throw SpikeError("The batched write operation needs at least one tensor.");
  }

  if (tensors.size() != data_objs.size() ||
      (offsets.has_value() && data_objs.size() != offsets.value().size())) {
    throw SpikeError("All parameters must be lists of same length.");
  }

  uint64_t batch_id = next_batch_id_++;

  std::vector<PreparedTensorWrite> prepared;
  prepared.reserve(tensors.size());

  uint32_t core_id = tensors[0]->get_core_id();

  for (size_t i = 0; i < tensors.size(); ++i) {
    std::shared_ptr<NrtTensor> tensor = std::move(tensors[i]);
    nb::ndarray<> data_obj = std::move(data_objs[i]);
    size_t offset = offsets.has_value() ? offsets.value()[i] : 0;

    const void *data = data_obj.data();
    size_t size = data_obj.nbytes();
    uint32_t tensor_core_id = tensor->get_core_id();
    if (core_id != tensor_core_id) {
      throw SpikeError("All tensors must be on the same NeuronCore.");
    }

    PreparedTensorWrite entry;
    entry.tensor = std::move(tensor);
    entry.data = data;
    entry.size = size;
    entry.offset = offset;
    entry.data_obj = std::move(data_obj);
    prepared.push_back(std::move(entry));
  }

  tensor_write_batched_prepared_[batch_id] = std::move(prepared);

  return batch_id;
}

uint64_t Spike::tensor_write_nonblock_batched_start(uint64_t batch_id) {
  auto it = tensor_write_batched_prepared_.find(batch_id);
  if (it == tensor_write_batched_prepared_.end()) {
    throw SpikeError("The batch ID does not exist.");
  }
  const std::vector<PreparedTensorWrite> &prepared = it->second;

  uint64_t cmd_id = next_non_block_id_++;
  uint32_t lnc = prepared[0].tensor->get_core_id();

  PendingTensorWriteBatched p;
  p.batch_id = batch_id;
  p.rets.assign(prepared.size(), NRT_SUCCESS);

  ensure_channel_registered(lnc, NRTA_XU_TENSOR_OP);
  // p.rets is a heap vector; the runtime keeps &p.rets[i] and fills it on
  // completion. Moving p into the deque below preserves that buffer, so the
  // pointers stay valid. All sub-requests hit the same (lnc, xu=TENSOR_OP,
  // queue=0) back-to-back, so seq numbers are consecutive and we only wait on
  // the last.
  nrta_seq_t wait_seq = 0;
  for (size_t i = 0; i < prepared.size(); ++i) {
    const PreparedTensorWrite &e = prepared[i];
    nrta_seq_t seq;
    NRT_STATUS status = nrta_tensor_write(
        e.tensor->get_ptr(), e.data, e.offset, e.size, static_cast<int>(lnc),
        /*queue=*/0, &p.rets[i], &seq);
    if (status != NRT_SUCCESS) {
      throw NrtError(status,
                     "Failed to schedule nonblocking batched tensor write");
    }
    wait_seq = seq;
  }

  enqueue_pending(lnc, NRTA_XU_TENSOR_OP,
                  PendingOp{cmd_id, wait_seq, std::move(p)});
  return cmd_id;
}

uint64_t Spike::tensor_read_nonblock_batched_prepare(
    std::vector<std::shared_ptr<const NrtTensor>> tensors,
    std::vector<nb::ndarray<>> dests,
    std::optional<std::vector<size_t>> offsets,
    std::optional<std::vector<size_t>> sizes) {
  if (tensors.size() == 0) {
    throw SpikeError("The batched read operation needs at least one tensor.");
  }

  if (tensors.size() != dests.size() ||
      (offsets.has_value() && dests.size() != offsets.value().size()) ||
      (sizes.has_value() && dests.size() != sizes.value().size())) {
    throw SpikeError("All parameters must be lists of same length.");
  }

  uint64_t batch_id = next_batch_id_++;

  std::vector<PreparedTensorRead> prepared;
  prepared.reserve(tensors.size());

  uint32_t core_id = tensors[0]->get_core_id();

  for (size_t i = 0; i < tensors.size(); ++i) {
    std::shared_ptr<const NrtTensor> tensor = std::move(tensors[i]);
    nb::ndarray<> dest = std::move(dests[i]);

    size_t offset = offsets.has_value() ? offsets.value()[i] : 0;
    size_t size = sizes.has_value() ? sizes.value()[i]
                                    : (tensor->get_size() - offset);

    if (dest.nbytes() < size) {
      throw SpikeError("The read operation exceeds the destination bound.");
    }

    void *data = dest.data();
    uint32_t tensor_core_id = tensor->get_core_id();
    if (core_id != tensor_core_id) {
      throw SpikeError("All tensors must be on the same NeuronCore.");
    }

    PreparedTensorRead entry;
    entry.tensor = std::move(tensor);
    entry.offset = offset;
    entry.size = size;
    entry.data = data;
    entry.data_obj = std::move(dest);
    prepared.push_back(std::move(entry));
  }

  tensor_read_batched_prepared_[batch_id] = std::move(prepared);

  return batch_id;
}

uint64_t Spike::tensor_read_nonblock_batched_start(uint64_t batch_id) {
  auto it = tensor_read_batched_prepared_.find(batch_id);
  if (it == tensor_read_batched_prepared_.end()) {
    throw SpikeError("The batch ID does not exist.");
  }
  const std::vector<PreparedTensorRead> &prepared = it->second;

  uint64_t cmd_id = next_non_block_id_++;
  uint32_t lnc = prepared[0].tensor->get_core_id();

  PendingTensorReadBatched p;
  p.batch_id = batch_id;
  p.rets.assign(prepared.size(), NRT_SUCCESS);

  ensure_channel_registered(lnc, NRTA_XU_TENSOR_OP);
  // p.rets is a heap vector; the runtime keeps &p.rets[i] and fills it on
  // completion. Moving p into the deque below preserves that buffer, so the
  // pointers stay valid. Seq numbers are consecutive; wait on the last.
  nrta_seq_t wait_seq = 0;
  for (size_t i = 0; i < prepared.size(); ++i) {
    const PreparedTensorRead &e = prepared[i];
    nrta_seq_t seq;
    NRT_STATUS status = nrta_tensor_read(
        e.data, e.tensor->get_ptr(), e.offset, e.size, static_cast<int>(lnc),
        /*queue=*/0, &p.rets[i], &seq);
    if (status != NRT_SUCCESS) {
      throw NrtError(status,
                     "Failed to schedule nonblocking batched tensor read");
    }
    wait_seq = seq;
  }

  enqueue_pending(lnc, NRTA_XU_TENSOR_OP,
                  PendingOp{cmd_id, wait_seq, std::move(p)});
  return cmd_id;
}

uint64_t Spike::execute_nonblock(
    std::shared_ptr<NrtModel> model,
    std::shared_ptr<const NrtTensorSet> input_set,
    std::shared_ptr<NrtTensorSet> output_set,
    std::optional<std::string> ntff_name, bool save_trace) {
  // nrta_execute_schedule has no profiling hook. The nrt_profile_start/stop
  // pair in the sync path straddles a single synchronous nrt_execute, which
  // can't be mapped safely onto an asynchronous schedule.
  if (save_trace) {
    throw SpikeError("save_trace is not supported with nonblocking execute.");
  }
  (void)ntff_name;

  uint64_t cmd_id = next_non_block_id_++;
  uint32_t lnc = model->get_core_id();

  PendingExecute p;
  p.ret = NRT_SUCCESS;
  p.model = model;
  p.input_set = input_set;
  p.output_set = output_set;

  ensure_channel_registered(lnc, NRTA_XU_COMPUTE);
  PendingOp &enq = enqueue_pending(
      lnc, NRTA_XU_COMPUTE, PendingOp{cmd_id, /*wait_seq=*/0, std::move(p)});
  PendingExecute &q = std::get<PendingExecute>(enq.op);
  NRT_STATUS status =
      nrta_execute_schedule(model->get_ptr(), input_set->get_ptr(),
                            output_set->get_ptr(), /*queue=*/0, &q.ret,
                            &enq.wait_seq);
  if (status != NRT_SUCCESS) {
    xu_queues_[lnc][NRTA_XU_COMPUTE].pop_back();
    throw NrtError(status, "Failed to schedule nonblocking execute");
  }

  return cmd_id;
}

std::optional<std::variant<SpikeError, NrtError>>
Spike::ret_to_err(NRT_STATUS ret) {
  if (ret != NRT_SUCCESS) {
    return NrtError(ret, "Nonblocking operation failed");
  }
  return std::nullopt;
}

std::optional<std::variant<SpikeError, NrtError>>
Spike::rets_to_err(const std::vector<NRT_STATUS> &rets) {
  for (NRT_STATUS r : rets) {
    if (r != NRT_SUCCESS) {
      return NrtError(r, "Nonblocking operation failed");
    }
  }
  return std::nullopt;
}

std::optional<NonBlockResult> Spike::try_poll() {
  // The runtime signals each channel's eventfd whenever its XU completes a
  // sequence. We ask epoll (non-blocking) which channels made progress and
  // only scan those — rather than probing all MAX_LNC * NRTA_XU_TYPE_NUM
  // channels every call. Registration is done before submission (see the
  // tensor_*/execute_nonblock helpers), so every completion is guaranteed to
  // signal epoll.
  if (epoll_fd_ >= 0) {
    struct epoll_event events[NUM_CHANNELS];
    int n;
    do {
      n = epoll_wait(epoll_fd_, events, NUM_CHANNELS, /*timeout=*/0);
    } while (n < 0 && errno == EINTR);
    if (n < 0) {
      throw SpikeError(std::string("epoll_wait failed: ") +
                       std::strerror(errno));
    }
    for (int i = 0; i < n; ++i) {
      uint32_t ch = events[i].data.u32;
      // Drain the eventfd so it only fires again on a future completion
      // (level-triggered). A single read resets the counter regardless of how
      // many completions it accumulated; nrta_get_sequence reports the latest.
      uint64_t counter;
      int fd = channel_event_fds_[ch];
      if (fd >= 0) {
        ssize_t rc;
        do {
          rc = read(fd, &counter, sizeof(counter));
        } while (rc < 0 && errno == EINTR);
      }
      // Enqueue for scanning, guarding against duplicates in O(1).
      if (!scan_channel_queued_[ch]) {
        scan_channel_queued_[ch] = true;
        scan_channels_.push_back(ch);
      }
    }
  }

  // Each (lnc, xu) queue is FIFO in submission order, and within it nrta_seq_t
  // values are monotonically increasing, so we only check the front op's
  // wait_seq against nrta_get_sequence's latest-completed seq for that channel.
  //
  // scan_channels_ is a round-robin FIFO: we pop the front channel, and if it
  // yields a completed op we re-queue it at the back before returning, so a
  // continuously-busy channel can't starve the others. A channel that is empty
  // or whose front is not yet complete is dropped (its eventfd will re-queue it
  // on the next completion).
  while (!scan_channels_.empty()) {
    uint32_t ch = scan_channels_.front();
    scan_channels_.pop_front();
    scan_channel_queued_[ch] = false;

    uint32_t lnc = ch / NRTA_XU_TYPE_NUM;
    uint32_t xu_idx = ch % NRTA_XU_TYPE_NUM;
    std::deque<PendingOp> &q = xu_queues_[lnc][xu_idx];

    if (q.empty()) {
      continue;
    }
    nrta_seq_t front_wait_seq = q.front().wait_seq;

    nrta_xu_t xu = static_cast<nrta_xu_t>(xu_idx);
    nrta_seq_t latest = 0;
    NRT_STATUS s = nrta_get_sequence(lnc, xu, /*queue=*/0, &latest);
    if (s != NRT_SUCCESS ||
        NRTA_SEQ_GET_SEQ_NUM(front_wait_seq) > NRTA_SEQ_GET_SEQ_NUM(latest)) {
      // Front not completed yet; stop scanning this channel until its eventfd
      // signals the next completion.
      continue;
    }

    PendingOp op = std::move(q.front());
    q.pop_front();

    // A single completion may have advanced the sequence past several queued
    // ops, so re-queue this channel at the back: the next try_poll() re-checks
    // its new front (and drops it if empty/not-ready) before trusting epoll.
    if (!scan_channel_queued_[ch]) {
      scan_channel_queued_[ch] = true;
      scan_channels_.push_back(ch);
    }

    uint64_t id = op.id;
    return std::visit(
        [id](auto &p) -> NonBlockResult {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, PendingTensorWrite>) {
            return NonBlockTensorWriteResult{id, ret_to_err(p.ret)};
          } else if constexpr (std::is_same_v<T, PendingTensorRead>) {
            return NonBlockTensorReadResult{id, std::move(p.data_obj),
                                            ret_to_err(p.ret)};
          } else if constexpr (std::is_same_v<T, PendingTensorWriteBatched>) {
            return NonBlockTensorWriteResult{id, rets_to_err(p.rets)};
          } else if constexpr (std::is_same_v<T, PendingTensorReadBatched>) {
            // Lifetime anchors stay in tensor_read_batched_prepared_[batch_id]
            // so the same prepared batch can be _start'd again. Callers read
            // the data via the dests array they passed to _prepare, so the
            // result's data field is left default-constructed.
            return NonBlockTensorReadResult{
                id, std::variant<nb::bytes, nb::ndarray<>>{},
                rets_to_err(p.rets)};
          } else {
            static_assert(std::is_same_v<T, PendingExecute>);
            return NonBlockExecResult{id, ret_to_err(p.ret)};
          }
        },
        op.op);
  }

  return std::nullopt;
}

NrtModel Spike::wrap_model(nrt_model_t *ptr) {
  uint32_t core_id = NrtRuntime::get_model_lnc(ptr);
  // FIXME: We have no way to get rank_id from the model using public NRT APIs.
  // Okay for now as it is never actually used.
  uint32_t rank_id = 0;
  uint32_t world_size = 1;
  // FIXME: Value of cc_enabled is fake, but no one uses it, so it is fine for
  // the purpose.
  return NrtModel(ptr, core_id, true, rank_id, world_size);
}

NrtTensor Spike::wrap_tensor(nrt_tensor_t *ptr) {
  uint32_t core_id = NrtRuntime::get_tensor_lnc(ptr);

  if (core_id == -1) {
    // CPU tensor wrapping not supported
    throw SpikeError("Wrapping a CPU tensor is not supported");
  }

  size_t size = NrtRuntime::get_tensor_size(ptr);
  // FIXME: get its real name when NRT provides such public API
  const char *name = "wrapped_tensor";
  return NrtTensor(ptr, core_id, size, name, this);
}

NrtTensorSet Spike::wrap_tensor_set(nrt_tensor_set_t *ptr) {
  return NrtTensorSet(ptr);
}

std::string Spike::dtype_to_string(nrt_dtype_t dtype) {
  switch (dtype) {
  case NRT_DTYPE_FLOAT32:
    return "float32";
  case NRT_DTYPE_FLOAT16:
    return "float16";
  case NRT_DTYPE_BFLOAT16:
    return "bfloat16";
  case NRT_DTYPE_INT8:
    return "int8";
  case NRT_DTYPE_UINT8:
    return "uint8";
  case NRT_DTYPE_INT16:
    return "int16";
  case NRT_DTYPE_UINT16:
    return "uint16";
  case NRT_DTYPE_INT32:
    return "int32";
  case NRT_DTYPE_UINT32:
    return "uint32";
  case NRT_DTYPE_INT64:
    return "int64";
  case NRT_DTYPE_UINT64:
    return "uint64";
  default:
    return "unknown";
  }
}

ModelTensorInfo Spike::get_tensor_info(NrtModel &model) {
  // Get tensor info from NRT
  nrt_tensor_info_array_t *tensor_info = model.get_tensor_info();

  ModelTensorInfo result;

  // Process each tensor
  for (uint64_t i = 0; i < tensor_info->tensor_count; ++i) {
    const nrt_tensor_info_t &info = tensor_info->tensor_array[i];

    TensorMetadata metadata;
    metadata.size = info.size;
    metadata.dtype = dtype_to_string(info.dtype);

    if (info.shape && info.ndim > 0) {
      const uint32_t max_reasonable_ndim = 64;
      if (info.ndim <= max_reasonable_ndim) {
        metadata.shape =
            std::vector<int64_t>(info.shape, info.shape + info.ndim);
      } else {
        throw SpikeError("Tensor has unreasonably large ndim: " +
                         std::to_string(info.ndim));
      }
    }

    std::string tensor_name(info.name);
    if (info.usage == NRT_TENSOR_USAGE_INPUT) {
      result.inputs[tensor_name] = std::move(metadata);
    } else if (info.usage == NRT_TENSOR_USAGE_OUTPUT) {
      result.outputs[tensor_name] = std::move(metadata);
    }
  }

  // Free tensor info
  NrtModel::free_tensor_info(tensor_info);

  return result;
}

} // namespace spike
