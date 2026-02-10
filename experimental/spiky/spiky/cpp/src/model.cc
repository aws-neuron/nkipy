// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spiky/model.h"

#include "spiky/errors.h"
#include "spiky/runtime.h"

#include <fstream>
#include <vector>

extern "C" {
#include <nrt/nrt_status.h>
}

namespace spiky {

Model::~Model() { Unload(); }

Model::Model(Model&& other) noexcept {
  *this = std::move(other);
}

Model& Model::operator=(Model&& other) noexcept {
  if (this != &other) {
    Unload();
    ptr_ = other.ptr_;
    neff_path_ = std::move(other.neff_path_);
    core_id_ = other.core_id_;
    is_collective_ = other.is_collective_;
    rank_id_ = other.rank_id_;
    world_size_ = other.world_size_;
    other.ptr_ = nullptr;
  }
  return *this;
}

void Model::LoadFromFile(const std::string& neff_path, uint32_t core_id,
                         bool cc_enabled, uint32_t rank_id, uint32_t world_size) {
  if (!Runtime::Global().IsInitialized()) {
    throw SpikyError("spiky: runtime not initialized (call spiky.init())");
  }
  Unload();

  neff_path_ = neff_path;
  core_id_ = core_id;
  is_collective_ = cc_enabled;
  rank_id_ = rank_id;
  world_size_ = world_size;

  std::ifstream file(neff_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw SpikyError("spiky: failed to open NEFF: " + neff_path);
  }
  size_t sz = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<char> data(sz);
  if (!file.read(data.data(), sz)) {
    throw SpikyError("spiky: failed to read NEFF: " + neff_path);
  }

  NRT_STATUS status;
  if (cc_enabled) {
    status = nrt_load_collectives(data.data(), sz, core_id,
                                  1, rank_id, world_size, &ptr_);
  } else {
    status = nrt_load(data.data(), sz, core_id, 1, &ptr_);
  }

  if (status != NRT_SUCCESS || !ptr_) {
    throw SpikyError("spiky: nrt_load failed, status=" + std::to_string(status));
  }
}

void Model::Unload() {
  if (ptr_) {
    nrt_unload(ptr_);
    ptr_ = nullptr;
  }
}

static std::string DTypeToString(nrt_dtype_t dtype) {
  switch (dtype) {
    case NRT_DTYPE_FLOAT16: return "float16";
    case NRT_DTYPE_BFLOAT16: return "bfloat16";
    case NRT_DTYPE_FLOAT32: return "float32";
    case NRT_DTYPE_INT8: return "int8";
    case NRT_DTYPE_INT16: return "int16";
    case NRT_DTYPE_INT32: return "int32";
    case NRT_DTYPE_INT64: return "int64";
    case NRT_DTYPE_UINT8: return "uint8";
    case NRT_DTYPE_UINT16: return "uint16";
    case NRT_DTYPE_UINT32: return "uint32";
    case NRT_DTYPE_UINT64: return "uint64";
    default: return "unknown";
  }
}

std::pair<std::vector<TensorInfo>, std::vector<TensorInfo>> Model::GetTensorInfo() const {
  std::vector<TensorInfo> ins;
  std::vector<TensorInfo> outs;
  if (!ptr_) return {ins, outs};

  nrt_tensor_info_array_t* info = nullptr;
  NRT_STATUS status = nrt_get_model_tensor_info(ptr_, &info);
  if (status != NRT_SUCCESS || !info) {
    return {ins, outs};
  }

  for (uint64_t i = 0; i < info->tensor_count; ++i) {
    const nrt_tensor_info_t& ti = info->tensor_array[i];
    TensorInfo out;
    out.name = ti.name ? std::string(ti.name) : "";
    out.dtype = DTypeToString(ti.dtype);
    out.size_bytes = static_cast<int64_t>(ti.size);
    out.shape.reserve(ti.ndim);
    for (uint32_t d = 0; d < ti.ndim; ++d) out.shape.push_back(static_cast<int64_t>(ti.shape[d]));
    out.is_input = (ti.usage == NRT_TENSOR_USAGE_INPUT);
    if (out.is_input) ins.push_back(std::move(out));
    else outs.push_back(std::move(out));
  }

  nrt_free_model_tensor_info(info);
  return {ins, outs};
}

}  // namespace spiky

