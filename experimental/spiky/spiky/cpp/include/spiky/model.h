// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
}

namespace spiky {

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;
  int64_t size_bytes{0};
  bool is_input{false};
};

class Model {
 public:
  Model() = default;
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&& other) noexcept;
  Model& operator=(Model&& other) noexcept;

  void LoadFromFile(const std::string& neff_path, uint32_t core_id,
                    bool cc_enabled = false, uint32_t rank_id = 0,
                    uint32_t world_size = 1);
  void Unload();

  nrt_model_t* Ptr() const { return ptr_; }
  uint32_t CoreId() const { return core_id_; }
  const std::string& Path() const { return neff_path_; }

  std::pair<std::vector<TensorInfo>, std::vector<TensorInfo>> GetTensorInfo() const;

 private:
  nrt_model_t* ptr_{nullptr};
  std::string neff_path_;
  uint32_t core_id_{0};
  bool is_collective_{false};
  uint32_t rank_id_{0};
  uint32_t world_size_{1};
};

}  // namespace spiky

