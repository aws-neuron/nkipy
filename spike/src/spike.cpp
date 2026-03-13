#include "spike.h"
#include <iostream>
#include <Python.h>

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
  // Shut down nonblock threads if initialized
  for (auto &q : exec_queues_) {
    q.emplace(NonBlockCloseCmd{});
  }
  for (auto &q : tensor_queues_) {
    q.emplace(NonBlockCloseCmd{});
  }
  for (auto &t : exec_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  for (auto &t : tensor_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }

  runtime_.reset();
  g_alive_spike_instance_exists = false;
  return 0;
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

void Spike::init_nonblock() {
  // Initialize nonblocking thread pools: one execution thread and one tensor
  // thread per physical NeuronCore. Uses total core count (not just visible)
  // to avoid visible-to-physical ID mapping. Extra threads remain blocked
  // with no overhead.

  int num_nc = NrtRuntime::get_total_nc_count();
  exec_queues_ = std::vector<LockedQueue<NonBlockExecOrCloseCmd, true>>(num_nc);
  tensor_queues_ = std::vector<LockedQueue<NonBlockTensorCmd, true>>(num_nc);

  for (int i = 0; i < num_nc; ++i) {
    exec_threads_.emplace_back([this, i]() { loop_execute(i); });
    tensor_threads_.emplace_back([this, i]() { loop_tensor(i); });
  }
}

uint64_t Spike::tensor_write_nonblock(std::shared_ptr<NrtTensor> tensor,
                                       nb::bytes data_obj,
                                       size_t offset) {
  uint64_t cmd_id = next_non_block_id_++;

  const void *data = data_obj.data();
  size_t size = data_obj.size();
  uint32_t core_id = tensor->get_core_id();

  NonBlockTensorWriteCmd cmd{cmd_id,        std::move(tensor), data,
                             size,          std::move(data_obj), offset};
  tensor_queues_[core_id].emplace(std::move(cmd));

  return cmd_id;
}

uint64_t Spike::tensor_write_nonblock(std::shared_ptr<NrtTensor> tensor,
                                       nb::ndarray<> data_obj,
                                       size_t offset) {
  uint64_t cmd_id = next_non_block_id_++;

  const void *data = data_obj.data();
  size_t size = data_obj.nbytes();
  uint32_t core_id = tensor->get_core_id();

  NonBlockTensorWriteCmd cmd{cmd_id,        std::move(tensor), data,
                             size,          std::move(data_obj), offset};
  tensor_queues_[core_id].emplace(std::move(cmd));

  return cmd_id;
}

uint64_t Spike::tensor_write_nonblock(std::shared_ptr<NrtTensor> tensor,
                                       const void *data, size_t size,
                                       size_t offset) {
  uint64_t cmd_id = next_non_block_id_++;

  uint32_t core_id = tensor->get_core_id();

  NonBlockTensorWriteCmd cmd;
  cmd.id = cmd_id;
  cmd.tensor = std::move(tensor);
  cmd.data = data;
  cmd.size = size;
  cmd.offset = offset;

  tensor_queues_[core_id].emplace(std::move(cmd));

  return cmd_id;
}

uint64_t Spike::tensor_read_nonblock(std::shared_ptr<const NrtTensor> tensor,
                                      size_t offset, size_t size) {
  size = (size == 0) ? (tensor->get_size() - offset) : size;

  uint64_t cmd_id = next_non_block_id_++;

  uint32_t core_id = tensor->get_core_id();

  nb::bytes dest(nullptr, size);
  void *data = PyBytes_AsString(dest.ptr());

  NonBlockTensorReadCmd cmd{cmd_id,        std::move(tensor), offset,
                            size,          data,              std::move(dest)};
  tensor_queues_[core_id].emplace(std::move(cmd));

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

  uint32_t core_id = tensor->get_core_id();

  void *data = dest.data();

  NonBlockTensorReadCmd cmd{cmd_id,        std::move(tensor), offset,
                            size,          data,              std::move(dest)};
  tensor_queues_[core_id].emplace(std::move(cmd));

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

  std::vector<NonBlockTensorWriteCmd> cmds;
  cmds.reserve(tensors.size());

  uint32_t core_id = tensors[0]->get_core_id();

  for (size_t i = 0; i < tensors.size(); ++i) {
    std::shared_ptr<NrtTensor> tensor = std::move(tensors[i]);
    nb::ndarray<> data_obj = std::move(data_objs[i]);
    size_t offset;
    if (offsets.has_value()) {
      offset = offsets.value()[i];
    } else {
      offset = 0;
    }

    const void *data = data_obj.data();
    size_t size = data_obj.nbytes();
    uint32_t tensor_core_id = tensor->get_core_id();
    if (core_id != tensor_core_id) {
      throw SpikeError("All tensors must be on the same NeuronCore.");
    }

    NonBlockTensorWriteCmd cmd{batch_id, std::move(tensor),   data,
                               size,     std::move(data_obj), offset};
    cmds.push_back(std::move(cmd));
  }

  std::scoped_lock lk(tensor_write_batched_cmds_mtx_);
  tensor_write_batched_cmds_[batch_id] = std::move(cmds);

  return batch_id;
}

uint64_t Spike::tensor_write_nonblock_batched_start(uint64_t batch_id) {
  uint64_t cmd_id = next_non_block_id_++;

  NonBlockTensorWriteBatchedCmd cmd;
  cmd.id = cmd_id;
  cmd.batch_id = batch_id;

  std::shared_lock lk(tensor_write_batched_cmds_mtx_);

  auto it = tensor_write_batched_cmds_.find(batch_id);
  if (it == tensor_write_batched_cmds_.end()) {
    lk.unlock();
    throw SpikeError("The batch ID does not exist.");
  }
  uint32_t core_id = it->second[0].tensor->get_core_id();
  lk.unlock();

  tensor_queues_[core_id].emplace(std::move(cmd));

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

  std::vector<NonBlockTensorReadCmd> cmds;
  cmds.reserve(tensors.size());

  uint32_t core_id = tensors[0]->get_core_id();

  for (size_t i = 0; i < tensors.size(); ++i) {
    std::shared_ptr<const NrtTensor> tensor = std::move(tensors[i]);
    nb::ndarray<> dest = std::move(dests[i]);

    size_t offset = offsets.has_value() ? offsets.value()[i] : 0;
    size_t size = sizes.has_value() ? sizes.value()[i] :
                  (tensor->get_size() - offset);

    if (dest.nbytes() < size) {
      throw SpikeError("The read operation exceeds the destination bound.");
    }

    void *data = dest.data();
    uint32_t tensor_core_id = tensor->get_core_id();
    if (core_id != tensor_core_id) {
      throw SpikeError("All tensors must be on the same NeuronCore.");
    }

    NonBlockTensorReadCmd cmd{batch_id, std::move(tensor), offset,
                              size,     data,              std::move(dest)};
    cmds.push_back(std::move(cmd));
  }

  std::scoped_lock lk(tensor_read_batched_cmds_mtx_);
  tensor_read_batched_cmds_[batch_id] = std::move(cmds);

  return batch_id;
}

uint64_t Spike::tensor_read_nonblock_batched_start(uint64_t batch_id) {
  uint64_t cmd_id = next_non_block_id_++;

  NonBlockTensorReadBatchedCmd cmd;
  cmd.id = cmd_id;
  cmd.batch_id = batch_id;

  std::shared_lock lk(tensor_read_batched_cmds_mtx_);

  auto it = tensor_read_batched_cmds_.find(batch_id);
  if (it == tensor_read_batched_cmds_.end()) {
    lk.unlock();
    throw SpikeError("The batch ID does not exist.");
  }
  uint32_t core_id = it->second[0].tensor->get_core_id();
  lk.unlock();

  tensor_queues_[core_id].emplace(std::move(cmd));

  return cmd_id;
}

uint64_t Spike::execute_nonblock(
    std::shared_ptr<NrtModel> model,
    std::shared_ptr<const NrtTensorSet> input_set,
    std::shared_ptr<NrtTensorSet> output_set,
    std::optional<std::string> ntff_name, bool save_trace) {
  uint64_t cmd_id = next_non_block_id_++;

  uint32_t core_id = model->get_core_id();

  NonBlockExecCmd cmd{cmd_id,           std::move(model), std::move(input_set),
                      std::move(output_set), ntff_name,        save_trace};
  exec_queues_[core_id].emplace(std::move(cmd));

  return cmd_id;
}

std::optional<NonBlockResult> Spike::try_poll() {
  std::optional<NonBlockInternalResult> internal_res = noti_queue_.try_pop();
  if (internal_res.has_value()) {
    return convert_internal_result(internal_res.value());
  } else {
    return std::nullopt;
  }
}

NrtModel Spike::wrap_model(nrt_model_t *ptr) {
  uint32_t core_id = ptr->start_vnc;
  uint32_t rank_id = ptr->gid;
  uint32_t world_size = ptr->vnc_count;
  // FIXME: Value of cc_enabled is fake, but no one uses it, so it is fine for
  // the purpose.
  return NrtModel(ptr, core_id, true, rank_id, world_size);
}

NrtTensor Spike::wrap_tensor(nrt_tensor_t *ptr) {
  if (ptr->sto->vtpb_idx == -1) {
    // CPU tensor wrapping not supported
    throw SpikeError("Wrapping a CPU tensor is not supported");
  }

  size_t size = ptr->_size;
  const char *name = ptr->name;
  // Use vtpb_idx as the core ID (this is the logical NC ID)
  uint32_t core_id = ptr->sto->vtpb_idx;
  return NrtTensor(ptr, core_id, size, name, this);
}

NrtTensorSet Spike::wrap_tensor_set(nrt_tensor_set_t *ptr) {
  return NrtTensorSet(ptr);
}

void Spike::loop_tensor(int core_id) {
  while (true) {
    NonBlockTensorCmd cmd_ = tensor_queues_[core_id].pop();

    if (std::holds_alternative<NonBlockTensorReadCmd>(cmd_)) {
      NonBlockTensorReadCmd &cmd = std::get<NonBlockTensorReadCmd>(cmd_);

      NonBlockTensorReadInternalResult res;

      try {
        cmd.tensor->read(cmd.data, cmd.size, cmd.offset);
        res.err = std::nullopt;
      } catch (SpikeError &err) {
        res.err = std::move(err);
      } catch (NrtError &err) {
        res.err = std::move(err);
      }

      res.id = cmd.id;
      res.tensor = std::move(cmd.tensor);
      res.data_obj = std::move(cmd.data_obj);

      noti_queue_.emplace(std::move(res));
    } else if (std::holds_alternative<NonBlockTensorWriteCmd>(cmd_)) {
      NonBlockTensorWriteCmd &cmd = std::get<NonBlockTensorWriteCmd>(cmd_);

      NonBlockTensorWriteInternalResult res;

      try {
        cmd.tensor->write(cmd.data, cmd.size, cmd.offset);
        res.err = std::nullopt;
      } catch (SpikeError &err) {
        res.err = std::move(err);
      } catch (NrtError &err) {
        res.err = std::move(err);
      }

      res.id = cmd.id;
      res.tensor = std::move(cmd.tensor);
      res.data_obj = std::move(cmd.data_obj);

      noti_queue_.emplace(std::move(res));
    } else if (std::holds_alternative<NonBlockTensorWriteBatchedCmd>(cmd_)) {
      NonBlockTensorWriteBatchedCmd &batched_cmd =
          std::get<NonBlockTensorWriteBatchedCmd>(cmd_);

      NonBlockTensorWriteInternalResult res;
      res.id = batched_cmd.id;

      std::shared_lock lk(tensor_write_batched_cmds_mtx_);

      auto &cmds = tensor_write_batched_cmds_[batched_cmd.batch_id];

      for (auto &cmd : cmds) {
        try {
          cmd.tensor->write(cmd.data, cmd.size, cmd.offset);
          res.err = std::nullopt;
        } catch (SpikeError &err) {
          res.err = std::move(err);
        } catch (NrtError &err) {
          res.err = std::move(err);
        }

        if (res.err.has_value()) {
          break;
        }
      }

      noti_queue_.emplace(std::move(res));
    } else if (std::holds_alternative<NonBlockTensorReadBatchedCmd>(cmd_)) {
      NonBlockTensorReadBatchedCmd &batched_cmd =
          std::get<NonBlockTensorReadBatchedCmd>(cmd_);

      NonBlockTensorReadInternalResult res;
      res.id = batched_cmd.id;

      std::shared_lock lk(tensor_read_batched_cmds_mtx_);

      auto &cmds = tensor_read_batched_cmds_[batched_cmd.batch_id];

      for (auto &cmd : cmds) {
        try {
          cmd.tensor->read(cmd.data, cmd.size, cmd.offset);
          res.err = std::nullopt;
        } catch (SpikeError &err) {
          res.err = std::move(err);
        } catch (NrtError &err) {
          res.err = std::move(err);
        }

        if (res.err.has_value()) {
          break;
        }
      }

      noti_queue_.emplace(std::move(res));
    } else {
      break;
    }
  }
}

void Spike::loop_execute(int core_id) {
  while (true) {
    NonBlockExecOrCloseCmd cmd_ = exec_queues_[core_id].pop();

    if (std::holds_alternative<NonBlockExecCmd>(cmd_)) {
      NonBlockExecCmd &cmd = std::get<NonBlockExecCmd>(cmd_);

      NonBlockExecInternalResult res;

      try {
        cmd.model->execute(*cmd.input_set, *cmd.output_set, cmd.ntff_name,
                           cmd.save_trace);
        res.err = std::nullopt;
      } catch (SpikeError &err) {
        res.err = std::move(err);
      } catch (NrtError &err) {
        res.err = std::move(err);
      }

      res.id = cmd.id;
      res.model = std::move(cmd.model);
      res.input_set = std::move(cmd.input_set);
      res.output_set = std::move(cmd.output_set);

      noti_queue_.emplace(std::move(res));
    } else {
      break;
    }
  }
}

// Convert InternalResult to Result in GIL context.
// Moves nanobind objects to Result, leaving shared_ptrs in InternalResult to be
// destroyed here (with GIL held). This avoids calling Python object destructors
// in worker threads, preventing GIL deadlocks and context switch overhead.
NonBlockResult Spike::convert_internal_result(NonBlockInternalResult &internal_res_) {
  if (std::holds_alternative<NonBlockTensorReadInternalResult>(internal_res_)) {
    NonBlockTensorReadInternalResult &internal_res = std::get<NonBlockTensorReadInternalResult>(internal_res_);
    NonBlockTensorReadResult res{internal_res.id, std::move(internal_res.data_obj), std::move(internal_res.err)};
    return res;
  } else if (std::holds_alternative<NonBlockTensorWriteInternalResult>(internal_res_)) {
    NonBlockTensorWriteInternalResult &internal_res = std::get<NonBlockTensorWriteInternalResult>(internal_res_);
    NonBlockTensorWriteResult res{internal_res.id, std::move(internal_res.err)};
    return res;
  } else {
    NonBlockExecInternalResult &internal_res = std::get<NonBlockExecInternalResult>(internal_res_);
    return NonBlockExecResult{internal_res.id, std::move(internal_res.err)};
  }
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
