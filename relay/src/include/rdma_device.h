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
