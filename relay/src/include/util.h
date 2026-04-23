#pragma once
#include <arpa/inet.h>
#include <glog/logging.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ifaddrs.h>
#include <sched.h>
#include <stdarg.h>

namespace uccl {

#define UCCL_LOG_EP VLOG(2) << "[Endpoint] "

#ifndef likely
#define likely(X) __builtin_expect(!!(X), 1)
#endif

#ifndef unlikely
#define unlikely(X) __builtin_expect(!!(X), 0)
#endif

static inline std::string FormatVarg(char const* fmt, va_list ap) {
  char* ptr = nullptr;
  int len = vasprintf(&ptr, fmt, ap);
  if (len < 0) return "<FormatVarg() error>";

  std::string ret(ptr, len);
  free(ptr);
  return ret;
}

[[maybe_unused]] static inline std::string Format(char const* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  const std::string s = FormatVarg(fmt, ap);
  va_end(ap);
  return s;
}

static inline std::string get_dev_ip(char const* dev_name) {
  struct ifaddrs* ifAddrStruct = NULL;
  struct ifaddrs* ifa = NULL;
  void* tmpAddrPtr = NULL;

  getifaddrs(&ifAddrStruct);

  for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr) {
      continue;
    }
    if (strncmp(ifa->ifa_name, dev_name, strlen(dev_name)) != 0) {
      continue;
    }
    if (ifa->ifa_addr->sa_family == AF_INET) {  // check it is IP4
      // is a valid IP4 Address
      tmpAddrPtr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
      char addressBuffer[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
      VLOG(5) << Format("%s IP Address %s\n", ifa->ifa_name, addressBuffer);
      return std::string(addressBuffer);
    } else if (ifa->ifa_addr->sa_family == AF_INET6) {  // check it is IP6
      // is a valid IP6 Address
      tmpAddrPtr = &((struct sockaddr_in6*)ifa->ifa_addr)->sin6_addr;
      char addressBuffer[INET6_ADDRSTRLEN];
      inet_ntop(AF_INET6, tmpAddrPtr, addressBuffer, INET6_ADDRSTRLEN);
      VLOG(5) << Format("%s IP Address %s\n", ifa->ifa_name, addressBuffer);
      return std::string(addressBuffer);
    }
  }
  if (ifAddrStruct != NULL) freeifaddrs(ifAddrStruct);
  return std::string();
}

inline int get_dev_numa_node(char const* dev_name) {
  std::string path =
      Format("/sys/class/infiniband/%s/device/numa_node", dev_name);
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open " << path;
    return -1;
  }

  std::string line;
  if (!std::getline(file, line)) {
    LOG(ERROR) << "Failed to read " << path;
    return -1;
  }

  auto numa_node = std::stoi(line);
  DCHECK(numa_node != -1) << "NUMA node is -1 for " << dev_name;
  return numa_node;
}

inline void pin_thread_to_numa(int numa_node) {
  std::string cpumap_path =
      Format("/sys/devices/system/node/node%d/cpulist", numa_node);
  std::ifstream cpumap_file(cpumap_path);
  if (!cpumap_file.is_open()) {
    LOG(ERROR) << "Failed to open " << cpumap_path;
    return;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  std::string line;
  std::getline(cpumap_file, line);

  // Parse CPU ranges like "0-3,7-11"
  std::stringstream ss(line);
  std::string range;
  while (std::getline(ss, range, ',')) {
    size_t dash = range.find('-');
    if (dash != std::string::npos) {
      // Handle range like "0-3"
      int start = std::stoi(range.substr(0, dash));
      int end = std::stoi(range.substr(dash + 1));
      for (int cpu = start; cpu <= end; cpu++) {
        CPU_SET(cpu, &cpuset);
      }
    } else {
      // Handle single CPU like "7"
      int cpu = std::stoi(range);
      CPU_SET(cpu, &cpuset);
    }
  }

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset)) {
    LOG(ERROR) << "Failed to set thread affinity to NUMA node " << numa_node;
  }
}

namespace fs = std::filesystem;

static inline std::string get_instance_type() {
  std::ifstream ifs("/sys/devices/virtual/dmi/id/product_name");
  std::string instance_type;
  if (ifs && std::getline(ifs, instance_type)) {
    // Trim trailing whitespace / newline
    while (!instance_type.empty() &&
           (instance_type.back() == '\n' || instance_type.back() == '\r' ||
            instance_type.back() == ' '))
      instance_type.pop_back();
    if (!instance_type.empty()) return instance_type;
  }
  return "unknown";
}

static inline std::vector<size_t> load_gpu_nic_map(
    std::string const& instance_type, int gpu_idx) {
  // Hardcoded GPU-NIC mappings for different instance types
  static const std::vector<std::vector<size_t>> p5_48xlarge = {
      {0, 1, 2, 3},     {4, 5, 6, 7},     {8, 9, 10, 11},   {12, 13, 14, 15},
      {16, 17, 18, 19}, {20, 21, 22, 23}, {24, 25, 26, 27}, {28, 29, 30, 31},
  };

  static const std::vector<std::vector<size_t>> p5en_48xlarge = {
      {0, 1},   {2, 3},   {4, 5},   {6, 7},
      {8, 9},   {10, 11}, {12, 13}, {14, 15},
  };

  static const std::vector<std::vector<size_t>> trn1_32xlarge = {
      {0}, {0}, {0}, {0}, {6}, {6}, {6}, {6}, {2}, {2}, {2},
      {2}, {4}, {4}, {4}, {4}, {3}, {3}, {3}, {3}, {5}, {5},
      {5}, {5}, {1}, {1}, {1}, {1}, {7}, {7}, {7}, {7},
  };

  const std::vector<std::vector<size_t>>* mapping = nullptr;
  if (instance_type == "p5.48xlarge") {
    mapping = &p5_48xlarge;
  } else if (instance_type == "p5en.48xlarge") {
    mapping = &p5en_48xlarge;
  } else if (instance_type == "trn1.32xlarge") {
    mapping = &trn1_32xlarge;
  } else {
    LOG(WARNING) << "[RDMA] Unknown instance type: " << instance_type;
    return {};
  }

  if (gpu_idx < 0 || gpu_idx >= static_cast<int>(mapping->size())) {
    LOG(WARNING) << "[RDMA] Invalid GPU index " << gpu_idx << " for instance "
                 << instance_type;
    return {};
  }

  LOG(INFO) << "[RDMA] Loaded GPU " << gpu_idx << " NIC map for "
            << instance_type;
  return (*mapping)[gpu_idx];
}

}  // namespace uccl