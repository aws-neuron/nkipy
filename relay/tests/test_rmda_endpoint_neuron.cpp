#include "../memory_allocator.h"
#include "../rdma_device.h"
#include "../rdma_endpoint.h"
#include "util/net.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <nrt/nrt.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <thread>

// ---------------------------------------------------------------------------
// Trainium1 RDMA Write Correctness Test — NeuronCore HBM to NeuronCore HBM
//
// Memory is allocated on NeuronCore HBM via nrt_tensor_allocate() with
// NRT_TENSOR_PLACEMENT_DEVICE. The returned VA is mmap'd into user space,
// so standard memset/memcpy and ibv_reg_mr all work directly on it.
//
// The test validates:
//   - NICEndpoint connection setup over EFA on Trainium1
//   - RDMA Write correctness (NeuronCore HBM -> NeuronCore HBM across nodes)
//   - Per-NeuronCore NIC selection via the trn1 NIC map
// ---------------------------------------------------------------------------

#define NRT_CHECK(call)                                                     \
  do {                                                                      \
    NRT_STATUS _s = (call);                                                 \
    if (_s != NRT_SUCCESS) {                                                \
      std::cerr << "NRT error at " << __FILE__ << ":" << __LINE__           \
                << " status=" << static_cast<int>(_s) << "\n";              \
      throw std::runtime_error("NRT call failed (status=" +                 \
                               std::to_string(static_cast<int>(_s)) + ")"); \
    }                                                                       \
  } while (0)

DEFINE_int32(nc_index, 0,
             "NeuronCore index (used for NIC selection and HBM allocation)");
DEFINE_uint64(rank_id, 0, "Local rank ID");
DEFINE_uint64(port, 19998, "Local port for OOB server");
DEFINE_uint64(remote_rank, 1, "Remote rank ID to connect to");
DEFINE_string(remote_ip, "", "Remote IP address");
DEFINE_uint64(remote_port, 19998, "Remote port number");
DEFINE_int32(iterations, 100, "Number of iterations for correctness test");
DEFINE_uint64(buffer_size, 1024 * 1024, "Buffer size in bytes (default 1MB)");

// Example usage (two trn1.32xlarge nodes, NeuronCore HBM to NeuronCore HBM):
//
//   Node A: ./test_rmda_endpoint_neuron --rank_id=0 --remote_ip=<B>
//   --remote_port=19998 --nc_index=0 Node B: ./test_rmda_endpoint_neuron
//   --rank_id=1 --remote_ip=<A> --remote_port=19998 --nc_index=0

// ---------------------------------------------------------------------------
// OOB sync channel (EpollServer + EpollClient, CPU-only).
// ---------------------------------------------------------------------------
class OOBSync {
 public:
  OOBSync(uint16_t listen_port, std::string const& remote_ip,
          uint16_t remote_port) {
    server_ = std::make_shared<EpollServer>(
        listen_port, [this](std::string const& /*input*/, std::string& output,
                            std::string const& /*ip*/, int /*port*/) {
          std::unique_lock<std::mutex> lk(mtx_);
          cv_.wait(lk, [this] { return ready_; });
          output = response_;
          ready_ = false;
        });
    assert(server_->start());

    client_ = std::make_shared<EpollClient>();
    assert(client_->start());
    while (conn_key_.empty()) {
      conn_key_ = client_->connect_to_server(remote_ip, remote_port);
      if (conn_key_.empty())
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }

  ~OOBSync() {
    if (client_) client_->stop();
    if (server_) server_->stop();
  }

  template <typename T>
  T exchange(T const& local) {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      response_.assign(reinterpret_cast<char const*>(&local), sizeof(T));
      ready_ = true;
    }
    cv_.notify_one();

    auto promise = std::make_shared<std::promise<std::string>>();
    auto future = promise->get_future();
    std::string payload(reinterpret_cast<char const*>(&local), sizeof(T));
    bool ok = client_->send_meta(
        conn_key_, payload,
        [promise](std::string const& resp) { promise->set_value(resp); });
    assert(ok);

    std::string resp = future.get();
    T remote{};
    std::memcpy(&remote, resp.data(), std::min(resp.size(), sizeof(T)));
    return remote;
  }

  void barrier() {
    char dummy = 'B';
    exchange(dummy);
  }

 private:
  std::shared_ptr<EpollServer> server_;
  std::shared_ptr<EpollClient> client_;
  std::string conn_key_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool ready_ = false;
  std::string response_;
};

// ---------------------------------------------------------------------------
// NeuronCore HBM memory helpers
// ---------------------------------------------------------------------------

// Allocate a RegMemBlock backed by NeuronCore HBM.
// nrt_tensor_allocate(DEVICE) places memory on the NeuronCore's HBM.
// nrt_tensor_get_va() returns a CPU-accessible mmap'd VA, so standard
// memset/memcpy and ibv_reg_mr all work directly on it.
static std::shared_ptr<RegMemBlock> alloc_neuron_hbm(int nc_index, size_t size,
                                                     char const* name) {
  nrt_tensor_t* tensor = nullptr;
  NRT_CHECK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, nc_index, size,
                                name, &tensor));
  void* va = nrt_tensor_get_va(tensor);
  if (!va) {
    nrt_tensor_free(&tensor);
    throw std::runtime_error(
        std::string("nrt_tensor_get_va returned NULL for ") + name);
  }

  std::memset(va, 0, size);

  std::cout << "  " << name << ": va=" << va << " size=" << size
            << " (NeuronCore " << nc_index << " HBM)\n";

  auto deleter = [tensor](RegMemBlock* b) mutable {
    if (tensor) nrt_tensor_free(&tensor);
    delete b;
  };
  return std::shared_ptr<RegMemBlock>(
      new RegMemBlock(va, size, MemoryType::GPU), deleter);
}

// Register a RegMemBlock with all RDMA contexts and copy MRs back into it.
// Uses standard ibv_reg_mr — works because the VA is mmap'd user-space memory.
static void regmr(NICEndpoint& ep, std::shared_ptr<RegMemBlock>& mem,
                  MRArray& mr_array) {
  assert(ep.uccl_regmr(mem->addr, mem->size, mr_array) == 0);
  for (uint32_t c = 0; c < kNICContextNumber; ++c)
    mem->setMRByContextID(c, mr_array.getKeyByContextID(c));
}

// ---------------------------------------------------------------------------
// Correctness test for RDMA Write (NeuronCore HBM -> NeuronCore HBM).
//
// Full write lifecycle per iteration:
//   1. Allocate & register NeuronCore HBM buffers (send + recv)
//   2. Connect (uccl_connect / uccl_accept)
//   3. Exchange RemoteMemInfo so writer knows target HBM address + rkeys
//   4. memset send buffer with deterministic pattern (VA is CPU-accessible)
//   5. Build RDMASendRequest (SendType::Write)
//   6. Post RDMA write via writeOrRead()  (NIC DMAs HBM-to-HBM over EFA)
//   7. Poll send completion via checkSendComplete_once()
//   8. Receiver reads recv buffer directly and verifies
//   9. Deregister memory
// ---------------------------------------------------------------------------
void correctness_write_test(NICEndpoint& endpoint) {
  size_t const bufsz = FLAGS_buffer_size;
  int const niters = FLAGS_iterations;
  bool const is_writer = (FLAGS_rank_id == 0);

  std::cout << "\n=== RDMA Write Correctness Test (" << niters << " iters, "
            << bufsz << " B, NeuronCore HBM) ===\n"
            << std::flush;

  // 1. Allocate NeuronCore HBM buffers and register with RDMA
  std::cout << "Allocating NeuronCore HBM buffers on NC " << FLAGS_nc_index
            << "...\n";
  auto send_mem = alloc_neuron_hbm(FLAGS_nc_index, bufsz, "send_buf");
  auto recv_mem = alloc_neuron_hbm(FLAGS_nc_index, bufsz, "recv_buf");

  MRArray send_mr, recv_mr;
  regmr(endpoint, send_mem, send_mr);
  regmr(endpoint, recv_mem, recv_mr);
  std::cout << "RDMA registration complete (ibv_reg_mr on mmap'd HBM).\n";

  // 2. Connection: rank 0 connects, rank 1 accepts
  ConnID conn_id;
  std::string peer_ip;
  int peer_ncidx = -1;

  if (is_writer) {
    conn_id = endpoint.uccl_connect(FLAGS_nc_index, FLAGS_remote_ip,
                                    static_cast<uint16_t>(FLAGS_remote_port));
  } else {
    conn_id = endpoint.uccl_accept(peer_ip, &peer_ncidx);
  }
  uint64_t peer_rank =
      static_cast<uint64_t>(reinterpret_cast<intptr_t>(conn_id.context));

  std::cout << "Rank " << FLAGS_rank_id
            << ": connected (flow_id=" << conn_id.flow_id << ")\n"
            << std::flush;

  std::this_thread::sleep_for(std::chrono::seconds(3));

  // 3. OOB sync channel for metadata exchange & per-iteration barriers.
  uint16_t sync_port = static_cast<uint16_t>(FLAGS_port + 1000);
  uint16_t remote_sync_port = static_cast<uint16_t>(FLAGS_remote_port + 1000);
  OOBSync sync(sync_port, FLAGS_remote_ip, remote_sync_port);

  // Exchange recv-buffer RemoteMemInfo so writer knows where to RDMA-write.
  RemoteMemInfo local_recv_info(recv_mem);
  RemoteMemInfo remote_recv_info = sync.exchange(local_recv_info);

  std::cout << "Rank " << FLAGS_rank_id << ": remote HBM addr=0x" << std::hex
            << remote_recv_info.addr << std::dec
            << " len=" << remote_recv_info.length << "\n"
            << std::flush;

  // 4-8. Iteration loop
  int passed = 0, failed = 0;

  for (int iter = 0; iter < niters; ++iter) {
    uint8_t pattern = static_cast<uint8_t>((iter + 1) & 0xFF);

    // Fill send buffer, clear recv buffer.
    // The VA is mmap'd NeuronCore HBM — CPU-accessible, so direct memset works.
    std::memset(send_mem->addr, pattern, bufsz);
    std::memset(recv_mem->addr, 0, bufsz);

    // Barrier: both sides have cleared their recv buffer
    sync.barrier();

    if (is_writer) {
      // Build & post RDMA Write
      auto req = std::make_shared<RDMASendRequest>(
          send_mem, std::make_shared<RemoteMemInfo>(remote_recv_info));
      req->send_type = SendType::Write;
      req->to_rank_id = static_cast<uint32_t>(peer_rank);

      int64_t wr_id = endpoint.writeOrRead(req);
      if (wr_id < 0) {
        std::cerr << "Iter " << iter << " FAIL: writeOrRead=" << wr_id << "\n";
        ++failed;
        continue;
      }

      // Poll send completion
      auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(30);
      while (!endpoint.checkSendComplete_once(peer_rank, wr_id)) {
        if (std::chrono::steady_clock::now() > deadline) {
          std::cerr << "Iter " << iter << " FAIL: send completion timeout\n";
          break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }

      if (std::chrono::steady_clock::now() > deadline) {
        ++failed;
        continue;
      }

      ++passed;
      if (iter % 10 == 0)
        std::cout << "Rank 0: iter " << iter << " done (wr_id=" << wr_id
                  << ")\n"
                  << std::flush;

    } else {
      // Wait for RDMA write to land in NeuronCore HBM recv buffer.
      // The mmap'd VA is CPU-visible; poll the first and last byte directly.
      auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(30);
      bool data_arrived = false;

      auto* recv_ptr = static_cast<uint8_t volatile*>(recv_mem->addr);

      while (!data_arrived) {
        uint8_t first = recv_ptr[0];
        uint8_t last = recv_ptr[bufsz - 1];
        if (first == pattern && last == pattern) {
          data_arrived = true;
        } else if (std::chrono::steady_clock::now() > deadline) {
          break;
        } else {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      }

      // Full verification: compare the entire recv buffer.
      bool ok = false;
      if (data_arrived) {
        std::vector<uint8_t> expected(bufsz, pattern);
        ok = (std::memcmp(recv_mem->addr, expected.data(), bufsz) == 0);
      }

      if (ok) {
        ++passed;
        if (iter % 10 == 0)
          std::cout << "Rank 1: iter " << iter << " PASS (0x" << std::hex
                    << (int)pattern << std::dec << ")\n"
                    << std::flush;
      } else {
        ++failed;
        std::cerr << "Iter " << iter << " FAIL on receiver\n";
      }
    }
  }

  // 9. Cleanup: deregister RDMA, then release RegMemBlocks (deleters call
  //    nrt_tensor_free to free the underlying NeuronCore HBM).
  endpoint.uccl_deregmr(send_mr);
  endpoint.uccl_deregmr(recv_mr);

  std::cout << "\n=== Results (Rank " << FLAGS_rank_id << ") ===\n"
            << "Passed: " << passed << "/" << niters << " ("
            << (100.0 * passed / niters) << "%)\n"
            << "Failed: " << failed << "\n"
            << std::flush;
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_minloglevel = google::WARNING;
  FLAGS_logtostderr = true;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_remote_ip.empty()) {
    std::cerr << "Error: --remote_ip is required!\n";
    return 1;
  }

  std::cout << "=== RDMA Write Correctness Test (NeuronCore HBM) ===\n"
            << "Rank=" << FLAGS_rank_id << "  NeuronCore=" << FLAGS_nc_index
            << "  Port=" << FLAGS_port << "\n"
            << "Remote: " << FLAGS_remote_ip << ":" << FLAGS_remote_port
            << "  Iters=" << FLAGS_iterations
            << "  BufSize=" << FLAGS_buffer_size << "\n"
            << "====================================================\n\n";

  try {
    // Initialize Neuron Runtime
    std::cout << "Initializing Neuron Runtime...\n";
    NRT_CHECK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
    std::cout << "Neuron Runtime initialized.\n";

    auto& dm = RdmaDeviceManager::instance();
    auto devs = dm.get_best_dev_idx(FLAGS_nc_index);
    std::cout << "Best EFA NICs for NeuronCore " << FLAGS_nc_index << ": ";
    for (auto d : devs) std::cout << d << " ";
    std::cout << "\n";

    NICEndpoint endpoint(FLAGS_nc_index, FLAGS_rank_id, FLAGS_port);
    std::cout << "NICEndpoint ready (port " << endpoint.get_p2p_listen_port()
              << ")\n";

    std::this_thread::sleep_for(std::chrono::seconds(2));
    correctness_write_test(endpoint);

    std::cout << "\nDone. Ctrl+C to exit.\n";
    while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
  } catch (std::exception const& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    nrt_close();
    return 1;
  }

  nrt_close();
  google::ShutdownGoogleLogging();
  return 0;
}
