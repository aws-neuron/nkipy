#pragma once
#include "define.h"
#include <cassert>
#include <memory>
#include <string>
#include <vector>

class RdmaDevice {
 public:
  RdmaDevice(struct ibv_device* dev)
      : dev_(dev), name_(ibv_get_device_name(dev)) {}

  std::string const& name() const { return name_; }

  std::shared_ptr<struct ibv_context> open() {
    struct ibv_context* ctx = ibv_open_device(dev_);
    if (!ctx) {
      perror("ibv_open_device failed");
      return nullptr;
    }
    return std::shared_ptr<struct ibv_context>(
        ctx, [](ibv_context* c) { ibv_close_device(c); });
  }

 private:
  struct ibv_device* dev_;
  std::string name_;
};

class RdmaDeviceManager {
 public:
  // Thread-safe singleton with C++11 static initialization
  // Automatically initializes on first call using std::call_once
  static RdmaDeviceManager& instance() {
    static RdmaDeviceManager inst;
    std::call_once(inst.init_flag_, &RdmaDeviceManager::initialize, &inst);
    return inst;
  }

  // Delete copy and move constructors/operators
  RdmaDeviceManager(RdmaDeviceManager const&) = delete;
  RdmaDeviceManager& operator=(RdmaDeviceManager const&) = delete;
  RdmaDeviceManager(RdmaDeviceManager&&) = delete;
  RdmaDeviceManager& operator=(RdmaDeviceManager&&) = delete;

  std::shared_ptr<RdmaDevice> getDevice(size_t id) {
    if (id >= devices_.size()) return nullptr;
    return devices_[id];
  }

  std::vector<size_t> get_best_dev_idx(int gpu_idx) {
    auto instance_type = relay::get_instance_type();
    auto selected_dev_indices = relay::load_gpu_nic_map(instance_type, gpu_idx);

    // If no hardcoded mapping, discover via NUMA topology
    if (selected_dev_indices.empty()) {
      selected_dev_indices = discover_nic_map_from_numa(gpu_idx);
    }

    CHECK(!selected_dev_indices.empty())
        << "[RDMA] GPU " << gpu_idx
        << " not found in GPU-NIC map for instance type: " << instance_type;

    std::stringstream ss;
    ss << "[RDMA] GPU " << gpu_idx << " selected NICs: ";
    for (size_t i = 0; i < selected_dev_indices.size(); i++) {
      ss << devices_[selected_dev_indices[i]]->name() << " (device idx "
         << selected_dev_indices[i] << ")";
      if (i < selected_dev_indices.size() - 1) ss << ", ";
    }
    LOG(INFO) << ss.str();

    return selected_dev_indices;
  }

  std::vector<size_t> discover_nic_map_from_numa(int gpu_idx) {
    // Find NUMA node of the NeuronCore/GPU
    int nc_numa = -1;
    // Try NeuronCore sysfs (Trn2 has 2 cores per device)
    for (int cores_per_dev : {2, 1}) {
      int neuron_dev = gpu_idx / cores_per_dev;
      std::string nc_path = relay::Format(
          "/sys/class/neuron_device/neuron%d/device/numa_node", neuron_dev);
      std::ifstream nc_file(nc_path);
      if (nc_file.is_open()) {
        std::string line;
        if (std::getline(nc_file, line)) {
          nc_numa = std::stoi(line);
          break;
        }
      }
    }

    // Collect NUMA nodes of all RDMA devices
    std::vector<size_t> same_numa, all_devs;
    for (size_t i = 0; i < devices_.size(); ++i) {
      all_devs.push_back(i);
      int dev_numa = relay::get_dev_numa_node(devices_[i]->name().c_str());
      if (nc_numa >= 0 && dev_numa == nc_numa) {
        same_numa.push_back(i);
      }
    }

    auto& result = same_numa.empty() ? all_devs : same_numa;

    // Distribute NICs across workers by rotating based on gpu_idx.
    // Each worker gets kNICContextNumber NICs starting at a different offset
    // to spread load across all available NICs.
    size_t n = result.size();
    size_t count = std::min(n, static_cast<size_t>(kNICContextNumber));
    size_t offset = (gpu_idx * count) % n;
    std::vector<size_t> selected;
    for (size_t i = 0; i < count; ++i) {
      selected.push_back(result[(offset + i) % n]);
    }

    LOG(INFO) << "[RDMA] Discovered NIC map for GPU " << gpu_idx
              << " (NUMA " << nc_numa << "): " << selected.size()
              << " NIC(s), offset " << offset;
    return selected;
  }

  int get_numa_node(size_t id) {
    if (id >= devices_.size()) {
      LOG(WARNING) << "Invalid device id: " << id;
      return -1;
    }
    std::string device_name = devices_[id]->name();
    return relay::get_dev_numa_node(device_name.c_str());
  }

 private:
  RdmaDeviceManager() = default;
  ~RdmaDeviceManager() = default;

  void initialize() {
    int num = 0;
    ibv_device** dev_list = ibv_get_device_list(&num);
    if (!dev_list) {
      perror("ibv_get_device_list failed");
      return;
    }
    std::cout << "RdmaDeviceManager: Found " << num << " RDMA device(s)"
              << std::endl;
    for (int i = 0; i < num; ++i) {
      auto dev = std::make_shared<RdmaDevice>(dev_list[i]);
      // std::cout << "  [" << i << "] " << dev->name() << std::endl;
      devices_.push_back(dev);
    }
    ibv_free_device_list(dev_list);
    std::cout << "RdmaDeviceManager: Initialization complete" << std::endl;
  }

  std::once_flag init_flag_;  // Ensures initialize() is called only once
  std::vector<std::shared_ptr<RdmaDevice>> devices_;
};
