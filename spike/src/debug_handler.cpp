#include "debug_handler.h"

#include <nrt/ndebug_stream.h>
#include <nrt/nrt.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sys/epoll.h>
#include <unistd.h>

namespace spike {

DebugHandler::DebugHandler(std::string output_dir)
    : output_dir_(std::move(output_dir)) {
  std::filesystem::create_directories(output_dir_);
}

DebugHandler::~DebugHandler() { stop(); }

void DebugHandler::connect() {
  uint32_t nc_count = 0;
  NRT_STATUS status = nrt_get_visible_nc_count(&nc_count);
  if (status != NRT_SUCCESS) {
    std::fprintf(stderr, "nrt_get_visible_nc_count failed: %d\n", status);
    return;
  }
  if (nc_count == 0) {
    std::fprintf(stderr, "No visible NeuronCores for debug stream\n");
    return;
  }

  for (uint32_t core_id = 0; core_id < nc_count; ++core_id) {
    int stream_fd = -1;
    status = nrt_debug_client_connect(core_id, &stream_fd);
    if (status != NRT_SUCCESS) {
      std::fprintf(stderr, "nrt_debug_client_connect(core=%u) failed: %d\n",
                   core_id, status);
      continue;
    }
    stream_fds_.push_back(stream_fd);
  }

  if (stream_fds_.empty()) {
    std::fprintf(stderr, "Failed to connect to any debug streams\n");
  }
}

void DebugHandler::start() {
  if (stream_fds_.empty())
    return;

  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ < 0) {
    std::fprintf(stderr, "epoll_create1 failed: %s\n", std::strerror(errno));
    return;
  }

  for (int fd : stream_fds_) {
    struct epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = fd;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
      std::fprintf(stderr, "epoll_ctl ADD failed for fd %d: %s\n", fd,
                   std::strerror(errno));
    }
  }

  stop_flag_.store(false);
  consumer_exited_.store(false);
  consumer_thread_ = std::thread(&DebugHandler::consumerLoop, this);
}

void DebugHandler::stop() {
  if (!consumer_thread_.joinable()) {
    // No thread running — just clean up FDs
    for (int fd : stream_fds_) {
      nrt_debug_client_connect_close(fd);
    }
    stream_fds_.clear();
    if (epoll_fd_ >= 0) {
      ::close(epoll_fd_);
      epoll_fd_ = -1;
    }
    return;
  }

  stop_flag_.store(true);

  // The consumer loop checks stop_flag_ every ~1s (epoll_wait timeout),
  // then drains remaining queued events (each involves disk I/O).
  // 30s budget: tiles are small (SBUF-bounded) but many may be queued.
  constexpr int kStopTimeoutMs = 30000;
  constexpr int kPollIntervalMs = 100;
  int waited_ms = 0;
  while (!consumer_exited_.load() && waited_ms < kStopTimeoutMs) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
    waited_ms += kPollIntervalMs;
  }

  if (consumer_exited_.load()) {
    consumer_thread_.join();
  } else {
    std::fprintf(
        stderr,
        "Warning: debug consumer thread did not stop within %ds, detaching. "
        "Leaking FDs to avoid use-after-close race.\n",
        kStopTimeoutMs / 1000);
    consumer_thread_.detach();
    // Don't close FDs — the detached thread may still be using them.
    // The OS reclaims them on process exit.
    stream_fds_.clear();
    epoll_fd_ = -1;
    return;
  }

  for (int fd : stream_fds_) {
    nrt_debug_client_connect_close(fd);
  }
  stream_fds_.clear();

  if (epoll_fd_ >= 0) {
    ::close(epoll_fd_);
    epoll_fd_ = -1;
  }
}

void DebugHandler::consumerLoop() {
  constexpr int MAX_EVENTS = 16;
  struct epoll_event events[MAX_EVENTS];

  while (!stop_flag_.load()) {
    int nfds = epoll_wait(epoll_fd_, events, MAX_EVENTS, 1000);
    if (nfds < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    for (int i = 0; i < nfds; ++i) {
      processEvent(events[i].data.fd);
    }
  }

  // Drain remaining events after stop is signaled
  for (;;) {
    int nfds = epoll_wait(epoll_fd_, events, MAX_EVENTS, 100);
    if (nfds <= 0)
      break;
    for (int i = 0; i < nfds; ++i) {
      processEvent(events[i].data.fd);
    }
  }

  consumer_exited_.store(true);
}

bool DebugHandler::processEvent(int stream_fd) {
  ndebug_stream_event_header_t header{};
  void *payload_raw = nullptr;

  NRT_STATUS status =
      nrt_debug_client_read_one_event(stream_fd, &header, &payload_raw);
  if (status != NRT_SUCCESS)
    return false;

  // RAII for the malloc'd payload
  std::unique_ptr<void, decltype(&std::free)> payload_guard(payload_raw,
                                                            std::free);

  if (payload_raw == nullptr)
    return false;

  if (header.type != NDEBUG_STREAM_EVENT_TYPE_DEBUG_TENSOR_READ)
    return false;

  auto *payload =
      static_cast<ndebug_stream_payload_debug_tensor_read_t *>(payload_raw);

  std::string prefix(payload->prefix);
  uint32_t logical_nc_id = payload->logical_nc_id;
  uint32_t pipe = payload->pipe;
  std::string dtype_str(payload->tensor_dtype);

  std::vector<uint64_t> shape;
  for (uint64_t dim : payload->tensor_shape) {
    if (dim > 0)
      shape.push_back(dim);
  }

  uint64_t tensor_data_size = payload->tensor_data_size;

  // Tensor data is inline via flexible array member
  const void *tensor_data = payload->tensor_data;

  saveTensor(prefix, logical_nc_id, pipe, dtype_str, shape, tensor_data_size,
             tensor_data);
  return true;
}

void DebugHandler::saveTensor(const std::string &prefix, uint32_t logical_nc_id,
                              uint32_t pipe, const std::string &dtype_str,
                              const std::vector<uint64_t> &shape,
                              uint64_t data_size, const void *tensor_data) {
  uint32_t iteration;
  {
    std::lock_guard<std::mutex> lock(iteration_mutex_);
    iteration = iteration_index_[prefix][logical_nc_id]++;
  }

  auto tensor_dir = output_dir_ / prefix /
                    ("core_" + std::to_string(logical_nc_id)) /
                    std::to_string(iteration);
  std::filesystem::create_directories(tensor_dir);

  // Build shape string
  std::string shape_str;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      shape_str += ", ";
    shape_str += std::to_string(shape[i]);
  }

  auto check_write = [&](std::ofstream &ofs, const char *filename) {
    if (!ofs.good()) {
      std::fprintf(stderr, "Warning: failed to write %s/%s\n",
                   tensor_dir.c_str(), filename);
    }
  };

  // metadata.txt
  {
    std::ofstream ofs(tensor_dir / "metadata.txt");
    ofs << "dtype = " << dtype_str << "\n";
    ofs << "shape = (" << shape_str << ")\n";
    ofs << "data_size = " << data_size << "\n";
    check_write(ofs, "metadata.txt");
  }

  // tensor_data.bin
  {
    std::ofstream ofs(tensor_dir / "tensor_data.bin", std::ios::binary);
    ofs.write(static_cast<const char *>(tensor_data), data_size);
    check_write(ofs, "tensor_data.bin");
  }

  // formatted_tensor.txt
  std::string formatted_all;
  std::string formatted_console;
  if (data_size > 0) {
    // Full output for file (all elements)
    formatted_all =
        formatTensor(tensor_data, data_size, dtype_str, shape, UINT32_MAX);
    // Abbreviated output for console
    formatted_console =
        formatTensor(tensor_data, data_size, dtype_str, shape, 1000);
  }

  {
    std::ofstream ofs(tensor_dir / "formatted_tensor.txt");
    if (!formatted_all.empty()) {
      ofs << formatted_all << "\n";
    } else {
      ofs << "Unknown dtype: " << dtype_str << "\n";
    }
    check_write(ofs, "formatted_tensor.txt");
  }

  // Print to stdout (pipe==0) or stderr (pipe!=0)
  auto rel_path = prefix + "/core_" + std::to_string(logical_nc_id) + "/" +
                  std::to_string(iteration);
  auto summary = prefix + ": dtype = " + dtype_str + ", shape = (" + shape_str +
                 "), data_size = " + std::to_string(data_size) +
                 " - Saved at " + rel_path;
  if (!formatted_console.empty()) {
    summary += " - formatted:\n" + formatted_console;
  }
  summary += "\n";

  FILE *output = (pipe != 0) ? stderr : stdout;
  std::fwrite(summary.data(), 1, summary.size(), output);
  std::fflush(output);
}

std::string DebugHandler::formatTensor(const void *data, size_t data_size,
                                       const std::string &dtype_str,
                                       const std::vector<uint64_t> &shape,
                                       uint32_t threshold) {
  TensorFormatOptions opts;
  opts.edge_items = 3;
  opts.threshold = threshold;
  opts.precision = 4;
  TensorFormatter formatter(opts);
  return formatter.format(data, data_size, dtype_str, shape);
}

} // namespace spike
