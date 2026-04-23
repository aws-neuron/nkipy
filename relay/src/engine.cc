#include "engine.h"
#include "endpoint_wrapper.h"
#include <arpa/inet.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>
#include <Python.h>
#include <sys/socket.h>
#include <unistd.h>

std::once_flag glog_init_once;
thread_local bool inside_python = false;

inline void check_python_signals() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  if (PyErr_CheckSignals() != 0) {
    std::cerr << "Python signal caught, exiting..." << std::endl;
    std::abort();
  }
  PyGILState_Release(gstate);
}

Endpoint::Endpoint(uint32_t const local_gpu_idx)
    : local_gpu_idx_(local_gpu_idx), passive_accept_(false) {
  std::cout << "Creating Engine with GPU index: " << local_gpu_idx << std::endl;

  std::call_once(glog_init_once,
                 []() { google::InitGoogleLogging("relay"); });
  FLAGS_minloglevel = parseLogLevelFromEnv();
  FLAGS_logtostderr = true;
  google::InstallFailureSignalHandler();

  // Initialize the RDMA endpoint.
  ep_ = std::shared_ptr<NICEndpoint>(
      new NICEndpoint(local_gpu_idx_, INVALID_RANK_ID, 0, false));

  std::cout << "Engine initialized for GPU " << local_gpu_idx_ << std::endl;
}

Endpoint::~Endpoint() {
  std::cout << "Destroying Engine..." << std::endl;

  stop_.store(true, std::memory_order_release);

  if (passive_accept_) {
    passive_accept_stop_.store(true, std::memory_order_release);
    if (passive_accept_thread_.joinable()) {
      passive_accept_thread_.join();
    }
  }

  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    for (auto& [conn_id, conn] : conn_id_to_conn_) {
      delete conn;
    }
  }

  {
    std::shared_lock<std::shared_mutex> lock(mr_mu_);
    for (auto& [mr_id, mr] : mr_id_to_mr_) {
      delete mr;
    }
  }

  std::cout << "Engine destroyed" << std::endl;
}

bool Endpoint::start_passive_accept() {
  if (!passive_accept_) {
    passive_accept_stop_.store(false, std::memory_order_release);
    passive_accept_thread_ =
        std::thread(&Endpoint::passive_accept_thread_func, this);
    passive_accept_ = true;
  }
  return true;
}

void Endpoint::initialize_engine() {
  auto const& actual_device_ids = ep_->get_device_ids();
  
  // Initialize rdma contexts for devices used by the GPU
  initialize_rdma_ctx_for_gpu(ep_, local_gpu_idx_);

  numa_node_ =
      RdmaDeviceManager::instance().get_numa_node(actual_device_ids[0]);

  std::cout << "Lazy creation of engine for GPU " << local_gpu_idx_
            << std::endl;
}

bool Endpoint::connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
                       uint64_t& conn_id) {
  std::cout << "Attempting to connect to " << ip_addr << ":" << remote_gpu_idx
            << " via port " << remote_port << std::endl;
  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  assert(local_gpu_idx_ != INVALID_GPU);

  std::future<ConnID> relay_conn_id_future = std::async(
      std::launch::async, [this, remote_gpu_idx, &ip_addr, remote_port]() {
        return relay_connect(ep_, remote_gpu_idx, ip_addr, remote_port);
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (relay_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  ConnID relay_conn_id =relay_conn_id_future.get();

  // Store the connection ID.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] =
        new Conn{conn_id, relay_conn_id, ip_addr, remote_gpu_idx};
  }
  return true;
}

std::vector<uint8_t> Endpoint::get_metadata() {
  std::string ip_str = relay::get_oob_ip();
  uint16_t port = get_p2p_listen_port(ep_);

  bool is_ipv6 = ip_str.find(':') != std::string::npos;
  size_t ip_len = is_ipv6 ? 16 : 4;

  // Additional 2 bytes for port and 4 bytes for local_gpu_idx_
  size_t total_len = ip_len + 2 + sizeof(int);
  std::vector<uint8_t> metadata(total_len);

  // Copy IP
  if (is_ipv6) {
    struct in6_addr ip6_bin;
    if (inet_pton(AF_INET6, ip_str.c_str(), &ip6_bin) != 1)
      throw std::runtime_error("Invalid IPv6 address: " + ip_str);
    std::memcpy(metadata.data(), &ip6_bin, 16);
  } else {
    struct in_addr ip4_bin;
    if (inet_pton(AF_INET, ip_str.c_str(), &ip4_bin) != 1)
      throw std::runtime_error("Invalid IPv4 address: " + ip_str);
    std::memcpy(metadata.data(), &ip4_bin, 4);
  }

  // Copy port in network byte order
  uint16_t net_port = htons(port);
  std::memcpy(metadata.data() + ip_len, &net_port, 2);

  // Copy local_gpu_idx_ in host byte order
  std::memcpy(metadata.data() + ip_len + 2, &local_gpu_idx_, sizeof(int));

  return metadata;
}

std::tuple<std::string, uint16_t, int> Endpoint::parse_metadata(
    std::vector<uint8_t> const& metadata) {
  if (metadata.size() == 10) {
    // IPv4: 4 bytes IP, 2 bytes port, 4 bytes GPU idx
    char ip_str[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, metadata.data(), ip_str, sizeof(ip_str)) ==
        nullptr) {
      throw std::runtime_error("Failed to parse IPv4 address from metadata");
    }

    uint16_t net_port;
    std::memcpy(&net_port, metadata.data() + 4, 2);
    uint16_t port = ntohs(net_port);

    int gpu_idx;
    std::memcpy(&gpu_idx, metadata.data() + 6, 4);

    return std::make_tuple(std::string(ip_str), port, gpu_idx);
  } else if (metadata.size() == 22) {
    // IPv6: 16 bytes IP, 2 bytes port, 4 bytes GPU idx
    char ip_str[INET6_ADDRSTRLEN];
    if (inet_ntop(AF_INET6, metadata.data(), ip_str, sizeof(ip_str)) ==
        nullptr) {
      throw std::runtime_error("Failed to parse IPv6 address from metadata");
    }

    uint16_t net_port;
    std::memcpy(&net_port, metadata.data() + 16, 2);
    uint16_t port = ntohs(net_port);

    int gpu_idx;
    std::memcpy(&gpu_idx, metadata.data() + 18, 4);

    return std::make_tuple(std::string(ip_str), port, gpu_idx);
  } else {
    throw std::runtime_error("Unexpected metadata length: " +
                             std::to_string(metadata.size()));
  }
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  std::cout << "Waiting to accept incoming connection..." << std::endl;

  // For demo purposes, simulate accepted connection
  conn_id = next_conn_id_.fetch_add(1);

  std::future<ConnID> relay_conn_id_future =
      std::async(std::launch::async, [this, &ip_addr, &remote_gpu_idx]() {
        return relay_accept(ep_, ip_addr, &remote_gpu_idx);
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (relay_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    if (passive_accept_ &&
        passive_accept_stop_.load(std::memory_order_acquire)) {
      std::cout << "Stop background accept..." << std::endl;
      stop_accept(ep_);
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  ConnID relay_conn_id =relay_conn_id_future.get();

  // Store the connection ID.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] =
        new Conn{conn_id, relay_conn_id, ip_addr, remote_gpu_idx};
  }

  return true;
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  mr_id = next_mr_id_.fetch_add(1);

  P2PMhandle* mhandle = new P2PMhandle();
  if (!relay_regmr(ep_, const_cast<void*>(data), size, mhandle)) {
    return false;
  }
  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    mr_id_to_mr_[mr_id] = new MR{mr_id, mhandle};
  }

  return true;
}

bool Endpoint::regv(std::vector<void const*> const& data_v,
                    std::vector<size_t> const& size_v,
                    std::vector<uint64_t>& mr_id_v) {
  if (data_v.size() != size_v.size())
    throw std::invalid_argument(
        "[Endpoint::regv] data_v/size_v length mismatch");

  size_t const n = data_v.size();

  // Early return if empty
  if (n == 0) {
    mr_id_v.clear();
    return true;
  }

  mr_id_v.resize(n);

  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    mr_id_to_mr_.reserve(mr_id_to_mr_.size() + n);
  }

  for (size_t i = 0; i < n; ++i) {
    uint64_t id = next_mr_id_.fetch_add(1);
    P2PMhandle* mhandle = new P2PMhandle();

    if (!relay_regmr(ep_, const_cast<void*>(data_v[i]), size_v[i], mhandle)) {
      std::cerr << "[Endpoint::regv] registration failed at i=" << i << '\n';
      return false;
    }

    {
      std::unique_lock<std::shared_mutex> lock(mr_mu_);
      mr_id_to_mr_[id] = new MR{id, mhandle};
    }
    mr_id_v[i] = id;
  }
  return true;
}

bool Endpoint::dereg(uint64_t mr_id) {
  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    auto it = mr_id_to_mr_.find(mr_id);
    if (it == mr_id_to_mr_.end()) {
      std::cerr << "[dereg] Error: Invalid mr_id " << mr_id << std::endl;
      return false;
    }
    auto mr = it->second;
    relay_deregmr(ep_, mr->mhandle_);
    delete mr;
    mr_id_to_mr_.erase(mr_id);
  }
  return true;
}

bool Endpoint::writev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                      std::vector<void*> src_v, std::vector<size_t> size_v,
                      std::vector<FifoItem> slot_item_v, size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[writev] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  RelayRequest ureq[kMaxInflightOps] = {};
  bool done[kMaxInflightOps] = {false};

  size_t iov_issued = 0, iov_finished = 0;

  while (iov_finished < num_iovs) {
    while (iov_issued < num_iovs &&
           iov_issued - iov_finished < kMaxInflightOps) {
      P2PMhandle* mhandle = get_mhandle(mr_id_v[iov_issued]);
      if (unlikely(mhandle == nullptr)) {
        std::cerr << "[writev] Error: Invalid mr_id " << mr_id_v[iov_issued]
                  << std::endl;
        return false;
      }

      auto rc = relay_write_async(ep_, conn, mhandle, src_v[iov_issued],
                                 size_v[iov_issued], slot_item_v[iov_issued],
                                 &ureq[iov_issued % kMaxInflightOps]);
      if (rc == -1) break;
      done[iov_issued % kMaxInflightOps] = false;
      iov_issued++;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;

    for (size_t i = iov_finished; i < iov_issued; i++) {
      if (done[i % kMaxInflightOps]) continue;
      if (relay_poll_once(ep_, &ureq[i % kMaxInflightOps])) {
        done[i % kMaxInflightOps] = true;
      }
    }

    while (iov_finished < iov_issued && done[iov_finished % kMaxInflightOps]) {
      iov_finished++;
    }
  }

  return true;
}

bool Endpoint::write(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
                     FifoItem const& slot_item) {
  DCHECK(size <= 0xffffffff) << "size must be < 4 GB";
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[write] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }
  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[write] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }
  RelayRequest ureq = {};
  FifoItem curr_slot_item = slot_item;
  curr_slot_item.size = size;

  while (relay_write_async(ep_, conn, mhandle, src, size, curr_slot_item,
                          &ureq) == -1)
    ;

  bool done = false;
  while (!done) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    if (relay_poll_once(ep_, &ureq)) {
      done = true;
    }
  }

  return true;
}

bool Endpoint::advertise(uint64_t conn_id, uint64_t mr_id, void* addr,
                         size_t len, char* out_buf) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[advertise] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }
  auto mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[advertise] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }
  if (prepare_fifo_metadata(ep_, conn, mhandle, addr, len, out_buf) == -1)
    return false;
  return true;
}

bool Endpoint::prepare_fifo(uint64_t mr_id, void* addr, size_t len,
                            char* out_buf) {
  auto mhandle = mr_id_to_mr_[mr_id]->mhandle_;
  // prepare_fifo_metadata doesn't actually use the endpoint or conn parameters
  if (prepare_fifo_metadata(ep_, nullptr, mhandle, addr, len, out_buf) == -1)
    return false;
  return true;
}

bool Endpoint::advertisev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                          std::vector<void*> addr_v, std::vector<size_t> len_v,
                          std::vector<char*> out_buf_v, size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[advertisev] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  std::vector<P2PMhandle*> mhandles(num_iovs);

  // Check if mhandles are all valid
  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[advertisev] Error: Invalid mr_id " << mr_id_v[i]
                << std::endl;
      return false;
    }
  }

  for (size_t i = 0; i < num_iovs; ++i) {
    auto mhandle = mhandles[i];
    if (prepare_fifo_metadata(ep_, conn, mhandle, addr_v[i], len_v[i],
                              out_buf_v[i]) == -1) {
      return false;
    }
  }
  return true;
}

bool Endpoint::add_remote_endpoint(std::vector<uint8_t> const& metadata,
                                   uint64_t& conn_id) {
  {
    // Check if we have connected to the remote endpoint before
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    auto it = remote_endpoint_to_conn_id_.find(metadata);
    if (it != remote_endpoint_to_conn_id_.end()) {
      conn_id = it->second;
      return true;
    }
  }
  auto [remote_ip, remote_port, remote_gpu_idx] = parse_metadata(metadata);
  bool success = connect(remote_ip, remote_gpu_idx, remote_port, conn_id);
  if (success) {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    remote_endpoint_to_conn_id_[metadata] = conn_id;
    return true;
  }
  return false;
}

void Endpoint::passive_accept_thread_func() {
  std::string ip_addr;
  int remote_gpu_idx;
  uint64_t conn_id;
  while (!stop_.load(std::memory_order_acquire)) {
    (void)accept(ip_addr, remote_gpu_idx, conn_id);
  }
}