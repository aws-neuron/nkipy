#ifndef SPIKE_SRC_INCLUDE_DEBUG_HANDLER_H
#define SPIKE_SRC_INCLUDE_DEBUG_HANDLER_H

#include "tensor_format.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace spike {

class DebugHandler {
public:
  explicit DebugHandler(std::string output_dir);
  ~DebugHandler();

  DebugHandler(const DebugHandler &) = delete;
  DebugHandler &operator=(const DebugHandler &) = delete;

  void connect();
  void start();
  void stop();

private:
  void consumerLoop();
  bool processEvent(int stream_fd);
  void saveTensor(const std::string &prefix, uint32_t logical_nc_id,
                  uint32_t pipe, const std::string &dtype_str,
                  const std::vector<uint64_t> &shape, uint64_t data_size,
                  const void *tensor_data);
  static std::string formatTensor(const void *data, size_t data_size,
                                  const std::string &dtype_str,
                                  const std::vector<uint64_t> &shape,
                                  uint32_t threshold);

  std::filesystem::path output_dir_;
  std::vector<int> stream_fds_;
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> consumer_exited_{false};
  std::thread consumer_thread_;
  int epoll_fd_{-1};

  std::mutex iteration_mutex_;
  // iteration_index_[prefix][logical_nc_id] -> next iteration number
  std::map<std::string, std::map<uint32_t, uint32_t>> iteration_index_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_DEBUG_HANDLER_H
