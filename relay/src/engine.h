#pragma once

#include "rdma_endpoint.h"
#include <glog/logging.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

extern thread_local bool inside_python;

int const kMaxNumGPUs = 8;

inline int parseLogLevelFromEnv() {
  char const* env = std::getenv("RELAY_LOG_LEVEL");
  if (!env) {
    return google::WARNING;
  }

  if (!strcasecmp(env, "INFO")) return google::INFO;
  if (!strcasecmp(env, "WARNING")) return google::WARNING;
  if (!strcasecmp(env, "ERROR")) return google::ERROR;
  if (!strcasecmp(env, "FATAL")) return google::FATAL;

  char* end = nullptr;
  long val = std::strtol(env, &end, 10);
  if (end != env && val >= 0 && val <= 3) {
    return static_cast<int>(val);
  }

  return google::WARNING;
}

using RDMAEndPoint = std::shared_ptr<NICEndpoint>;
enum ReqType { ReqTx, ReqRx, ReqWrite };
struct RelayRequest {
  enum ReqType type;
  uint32_t n;
  uint32_t engine_idx;
};
struct P2PMhandle {
  MRArray mr_array;
};

struct MR {
  uint64_t mr_id_;
  P2PMhandle* mhandle_;
};

struct Conn {
  uint64_t conn_id_;
  ConnID relay_conn_id_;
  std::string ip_addr_;
  int remote_gpu_idx_;
};

// Custom hash function for std::vector<uint8_t>
struct VectorUint8Hash {
  std::size_t operator()(std::vector<uint8_t> const& vec) const {
    std::size_t hash = vec.size();
    for (uint8_t byte : vec) {
      hash = hash * 31 + static_cast<std::size_t>(byte);
    }
    return hash;
  }
};

class Endpoint {
  static constexpr int kMaxInflightOps = 8;  // Max 8 concurrent Ops

 public:
  /* Create engine threads running in background for a single interface. It also
   * opens a TCP listening thread waiting for incoming connections. */
  Endpoint(uint32_t const local_gpu_idx);

  ~Endpoint();

  /* Connect to a remote server via TCP, then build RDMA QP connections. */
  bool connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
               uint64_t& conn_id);

  std::vector<uint8_t> get_metadata();

  /* Parse endpoint metadata to extract IP address, port, and GPU index.
   * Returns a tuple of (ip_address, port, gpu_index). */
  static std::tuple<std::string, uint16_t, int> parse_metadata(
      std::vector<uint8_t> const& metadata);

  /* Accept an incoming connection via TCP, then build RDMA QP connections. */
  bool accept(std::string& ip_addr, int& remote_gpu_idx, uint64_t& conn_id);

  /* Register the data with a specific interface. */
  bool reg(void const* data, size_t size, uint64_t& mr_id);

  bool regv(std::vector<void const*> const& data_v,
            std::vector<size_t> const& size_v, std::vector<uint64_t>& mr_id_v);
  bool dereg(uint64_t mr_id);

  /* Write data to the remote server. Blocking. */
  bool write(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
             FifoItem const& slot_item);

  /* Write a vector of data chunks. */
  bool writev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
              std::vector<void*> src_v, std::vector<size_t> size_v,
              std::vector<FifoItem> slot_item_v, size_t num_iovs);

  bool advertise(uint64_t conn_id, uint64_t mr_id, void* addr, size_t len,
                 char* out_buf);

  /* Prepare Fifo without requiring a connection (for pre-computing fifo_item).
   */
  bool prepare_fifo(uint64_t mr_id, void* addr, size_t len, char* out_buf);

  /* Advertise a vector of data chunks. */
  bool advertisev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                  std::vector<void*> addr_v, std::vector<size_t> len_v,
                  std::vector<char*> out_buf_v, size_t num_iovs);

  /* Add a remote endpoint with metadata - connect only once per remote
   * endpoint. */
  bool add_remote_endpoint(std::vector<uint8_t> const& metadata,
                           uint64_t& conn_id);

  /* Start a background thread for accepting. */
  bool start_passive_accept();

  /** Returns conn_id for @rank, or UINT64_MAX if unknown. */
  uint64_t conn_id_of_rank(int rank) const {
    auto it = rank2conn_.find(rank);
    return it != rank2conn_.end() ? it->second : UINT64_MAX;
  }

  inline MR* get_mr(uint64_t mr_id) const {
    std::shared_lock<std::shared_mutex> lock(mr_mu_);
    auto it = mr_id_to_mr_.find(mr_id);
    if (it == mr_id_to_mr_.end()) {
      return nullptr;
    }
    return it->second;
  }

  inline P2PMhandle* get_mhandle(uint64_t mr_id) const {
    auto mr = get_mr(mr_id);
    if (unlikely(mr == nullptr)) {
      return nullptr;
    }
    return mr->mhandle_;
  }

  inline Conn* get_conn(uint64_t conn_id) const {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = conn_id_to_conn_.find(conn_id);
    if (it == conn_id_to_conn_.end()) {
      return nullptr;
    }
    return it->second;
  }

 private:
  /* Initialize the engine Internal helper function for lazy initialization. */
  void initialize_engine();

  int local_gpu_idx_;
  int numa_node_;

  RDMAEndPoint ep_;

  std::atomic<uint64_t> next_conn_id_ = 0;
  std::atomic<uint64_t> next_mr_id_ = 0;
  // Accessed by both app thread and proxy thread.
  mutable std::shared_mutex conn_mu_;
  std::unordered_map<uint64_t, Conn*> conn_id_to_conn_;
  mutable std::shared_mutex mr_mu_;
  std::unordered_map<uint64_t, MR*> mr_id_to_mr_;

  std::unordered_map<std::vector<uint8_t>, uint64_t, VectorUint8Hash>
      remote_endpoint_to_conn_id_;

  // Single-threaded.
  std::unordered_map<int, uint64_t> rank2conn_;

  std::atomic<bool> stop_{false};

  std::atomic<bool> passive_accept_stop_{false};
  bool passive_accept_;
  std::thread passive_accept_thread_;
  void passive_accept_thread_func();
};
