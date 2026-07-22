#include "model.h"
#include "spike.h"

#include <cerrno>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <sys/eventfd.h>
#include <unistd.h>

namespace spike {

// AsyncExecution implementation
AsyncExecution::AsyncExecution(NrtTensorSet &&inputs, NrtTensorSet &&outputs,
                               nrta_seq_t sequence,
                               std::unique_ptr<NRT_STATUS> ret)
    : input_set_(std::move(inputs)), output_set_(std::move(outputs)),
      sequence_(sequence), ret_(std::move(ret)), waited_(false) {}

bool AsyncExecution::is_completed() const {
  bool completed = false;
  NRT_STATUS status = nrta_is_completed(sequence_, &completed);
  // nrta_is_completed returns NRT_INVALID only when the sequence id itself is
  // not valid; a valid-but-pending request reports completed=false. Treat an
  // invalid sequence as an error rather than silently reporting "not done".
  if (status == NRT_INVALID) {
    throw NrtError(status, "Invalid sequence id while polling async completion");
  }
  return completed;
}

void AsyncExecution::wait() {
  if (waited_) {
    return;
  }
  waited_ = true;

  // Block on an eventfd that NRT signals when the sequence completes. This
  // avoids busy-polling a CPU core (each TP rank is its own process and needs
  // its core for the host-side dispatch of the next token).
  int efd = eventfd(0, 0);
  if (efd < 0) {
    throw SpikeError("Failed to create eventfd for async execution wait");
  }

  // Registering an already-completed sequence signals the fd immediately, so
  // there is no lost-wakeup race between schedule and wait.
  NRT_STATUS status = nrta_event_register_seq_id_completion(sequence_, efd);
  if (status != NRT_SUCCESS) {
    ::close(efd);
    throw NrtError(status, "Failed to register async completion event");
  }

  uint64_t counter = 0;
  ssize_t n = ::read(efd, &counter, sizeof(counter));
  int read_errno = errno;

  // Deregister before closing so NRT drops its reference to the fd.
  nrta_event_register_seq_id_completion(sequence_, -1);
  ::close(efd);

  if (n != static_cast<ssize_t>(sizeof(counter))) {
    throw SpikeError("Failed to wait on async completion eventfd (errno=" +
                     std::to_string(read_errno) + ")");
  }

  // *ret_ holds the NRT_STATUS the execution reported on completion.
  if (ret_ && *ret_ != NRT_SUCCESS) {
    throw NrtError(*ret_, "Asynchronous model execution failed");
  }
}

// NrtModel implementation
NrtModel::NrtModel(const std::string &neff_path, uint32_t core_id,
                   bool cc_enabled, uint32_t rank_id, uint32_t world_size,
                   const Spike *spike)
    : ptr_(nullptr), neff_path_(neff_path), core_id_(core_id),
      rank_id_(rank_id), world_size_(world_size), is_collective_(cc_enabled),
      runtime_closed_(spike->get_runtime_closed_state()) {

  // Read NEFF file
  std::ifstream file(neff_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw SpikeError(
        "Failed to open NEFF file: " + neff_path +
        ". Please check if the file exists and have read permission.");
  }

  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> neff_data(file_size);
  if (!file.read(neff_data.data(), file_size)) {
    file.close();
    throw SpikeError(
        "Failed to read NEFF file: " + neff_path +
        ". Please check if the file exists and have read permission.");
  }
  file.close();

  // Load model
  NRT_STATUS status;
  if (cc_enabled) {
    status = nrt_load_collectives(
        neff_data.data(), file_size, core_id,
        1, // N.B.: the vnc_count is hardcoded to be 1;
           // initializing with vnc_count > 1 should have been deprecated;
        rank_id, world_size, &ptr_);
  } else {
    status = nrt_load(neff_data.data(), file_size, core_id,
                      1, // N.B.: same as above
                      &ptr_);
  }

  if (status != 0) {
    throw NrtError(status, "Failed to load model");
  }
}

NrtModel::NrtModel(nrt_model_t *ptr, uint32_t core_id, bool cc_enabled,
                   uint32_t rank_id, uint32_t world_size)
    : ptr_(ptr), core_id_(core_id), rank_id_(rank_id), world_size_(world_size),
      is_collective_(cc_enabled), runtime_closed_(nullptr) {}

NrtModel::~NrtModel() { unload(); }

NrtModel::NrtModel(NrtModel &&other) noexcept
    : ptr_(other.ptr_), neff_path_(std::move(other.neff_path_)),
      core_id_(other.core_id_), rank_id_(other.rank_id_),
      world_size_(other.world_size_), is_collective_(other.is_collective_),
      runtime_closed_(std::move(other.runtime_closed_)) {
  other.ptr_ = nullptr;
}

NrtModel &NrtModel::operator=(NrtModel &&other) noexcept {
  if (this != &other) {
    unload();
    ptr_ = other.ptr_;
    neff_path_ = std::move(other.neff_path_);
    core_id_ = other.core_id_;
    rank_id_ = other.rank_id_;
    world_size_ = other.world_size_;
    is_collective_ = other.is_collective_;
    runtime_closed_ = std::move(other.runtime_closed_);
    other.ptr_ = nullptr;
  }
  return *this;
}

bool NrtModel::is_unloaded() const {
  return ptr_ == nullptr || (is_owner() && runtime_closed_->load());
}

void NrtModel::execute(const NrtTensorSet &inputs, NrtTensorSet &outputs,
                       std::optional<std::string> ntff_name, bool save_trace) {
  if (is_unloaded()) {
    throw SpikeError("Unable to execute an unloaded model. Please check the "
                     "lifetime of the model.");
  }

  NRT_STATUS status;
  if (save_trace) {
    std::string actual_ntff_name = ntff_name.value_or("profile.ntff");
    nrt_profile_start(ptr_, actual_ntff_name.c_str());
    status = nrt_execute(ptr_, inputs.get_ptr(), outputs.get_ptr());
    nrt_profile_stop(actual_ntff_name.c_str());
  } else {
    status = nrt_execute(ptr_, inputs.get_ptr(), outputs.get_ptr());
  }

  if (status) {
    throw NrtError(status, "Failed to execute model");
  }
}

AsyncExecution NrtModel::execute_schedule(NrtTensorSet &&inputs,
                                          NrtTensorSet &&outputs) {
  if (is_unloaded()) {
    throw SpikeError("Unable to execute an unloaded model. Please check the "
                     "lifetime of the model.");
  }

  // `ret` receives the execution's completion status asynchronously, so it must
  // outlive the scheduled request; the returned AsyncExecution owns it. The
  // queue is fixed at 0 per the nrt_async.h contract for nrta_execute_schedule.
  auto ret = std::make_unique<NRT_STATUS>(NRT_SUCCESS);
  nrta_seq_t sequence = 0;
  NRT_STATUS status =
      nrta_execute_schedule(ptr_, inputs.get_ptr(), outputs.get_ptr(),
                            /*queue=*/0, ret.get(), &sequence);
  if (status != NRT_SUCCESS) {
    throw NrtError(status, "Failed to schedule asynchronous model execution");
  }

  return AsyncExecution(std::move(inputs), std::move(outputs), sequence,
                        std::move(ret));
}

nrt_tensor_info_array_t *NrtModel::get_tensor_info() {
  if (is_unloaded()) {
    throw SpikeError("Unable to get model tensor info from an unloaded model. "
                     "Please check the lifetime of the model.");
  }

  nrt_tensor_info_array_t *info = nullptr;
  NRT_STATUS status = nrt_get_model_tensor_info(ptr_, &info);
  if (status != 0) {
    throw NrtError(status, "Failed to get tensor info");
  }
  return info;
}

void NrtModel::free_tensor_info(nrt_tensor_info_array_t *info) {
  NRT_STATUS status = nrt_free_model_tensor_info(info);
  if (status != 0) {
    throw NrtError(status, "Failed to free tensor info");
  }
}

void NrtModel::unload() {
  if (is_unloaded() || !is_owner()) {
    return;
  }

  nrt_unload(ptr_);
  ptr_ = nullptr;
}

std::string NrtModel::to_string() const {
  std::ostringstream oss;
  oss << "NrtModel(ptr=" << ptr_ << ", core_id=" << core_id_
      << ", rank_id=" << rank_id_ << ", world_size=" << world_size_
      << ", is_collective=" << (is_collective_ ? "True" : "False")
      << ", is_owner=" << (is_owner() ? "True" : "False") << ")";
  return oss.str();
}

} // namespace spike
