// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "model.h"
#include "spike.h"

#include <fstream>
#include <sstream>

namespace spike {

// NrtModel implementation
NrtModel::NrtModel(const std::string &neff_path, uint32_t core_id,
                   bool cc_enabled, uint32_t rank_id, uint32_t world_size,
                   const Spike *spike)
    : ptr_(nullptr), neff_path_(neff_path), core_id_(core_id),
      rank_id_(rank_id), world_size_(world_size), is_collective_(cc_enabled),
      spike_(spike) {

  // Read NEFF file
  std::ifstream file(neff_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw SpikeError(
        "Failed to open NEFF file: " + neff_path +
        ". Please check if the file exists and have read permission.");
  }

  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> neff_data(file_size);
  if (!file.read(neff_data.data(), file_size)) {
    file.close();
    throw SpikeError(
        "Failed to read NEFF file: " + neff_path +
        ". Please check if the file exists and have read permission.");
  }
  file.close();

  // Load model
  NRT_STATUS status;
  if (cc_enabled) {
    status = nrt_load_collectives(
        neff_data.data(), file_size, core_id,
        1, // N.B.: the vnc_count is hardcoded to be 1;
           // initializing with vnc_count > 1 should have been deprecated;
        rank_id, world_size, &ptr_);
  } else {
    status = nrt_load(neff_data.data(), file_size, core_id,
                      1, // N.B.: same as above
                      &ptr_);
  }

  if (status != 0) {
    throw NrtError(status, "Failed to load model");
  }
}

NrtModel::NrtModel(nrt_model_t *ptr, uint32_t core_id, bool cc_enabled,
                   uint32_t rank_id, uint32_t world_size)
    : ptr_(ptr), core_id_(core_id), rank_id_(rank_id), world_size_(world_size),
      is_collective_(cc_enabled), spike_(nullptr) {}

NrtModel::~NrtModel() { unload(); }

NrtModel::NrtModel(NrtModel &&other) noexcept
    : ptr_(other.ptr_), neff_path_(std::move(other.neff_path_)),
      core_id_(other.core_id_), rank_id_(other.rank_id_),
      world_size_(other.world_size_), is_collective_(other.is_collective_),
      spike_(other.spike_) {
  other.ptr_ = nullptr;
}

NrtModel &NrtModel::operator=(NrtModel &&other) noexcept {
  if (this != &other) {
    unload();
    ptr_ = other.ptr_;
    neff_path_ = std::move(other.neff_path_);
    core_id_ = other.core_id_;
    rank_id_ = other.rank_id_;
    world_size_ = other.world_size_;
    is_collective_ = other.is_collective_;
    spike_ = other.spike_;
    other.ptr_ = nullptr;
  }
  return *this;
}

bool NrtModel::is_unloaded() const {
  return ptr_ == nullptr || (is_owner() && spike_->is_closed());
}

void NrtModel::execute(const NrtTensorSet &inputs, NrtTensorSet &outputs,
                       std::optional<std::string> ntff_name, bool save_trace) {
  if (is_unloaded()) {
    throw SpikeError("Unable to execute an unloaded model. Please check the "
                     "lifetime of the model.");
  }

  NRT_STATUS status;
  if (save_trace) {
    std::string actual_ntff_name = ntff_name.value_or("profile.ntff");
    nrt_profile_start(ptr_, actual_ntff_name.c_str());
    status = nrt_execute(ptr_, inputs.get_ptr(), outputs.get_ptr());
    nrt_profile_stop(actual_ntff_name.c_str());
  } else {
    status = nrt_execute(ptr_, inputs.get_ptr(), outputs.get_ptr());
  }

  if (status) {
    throw NrtError(status, "Failed to execute model");
  }
}

nrt_tensor_info_array_t *NrtModel::get_tensor_info() {
  if (is_unloaded()) {
    throw SpikeError("Unable to get model tensor info from an unloaded model. "
                     "Please check the lifetime of the model.");
  }

  nrt_tensor_info_array_t *info = nullptr;
  NRT_STATUS status = nrt_get_model_tensor_info(ptr_, &info);
  if (status != 0) {
    throw NrtError(status, "Failed to get tensor info");
  }
  return info;
}

void NrtModel::free_tensor_info(nrt_tensor_info_array_t *info) {
  NRT_STATUS status = nrt_free_model_tensor_info(info);
  if (status != 0) {
    throw NrtError(status, "Failed to free tensor info");
  }
}

void NrtModel::unload() {
  if (is_unloaded() || !is_owner()) {
    return;
  }

  nrt_unload(ptr_);
  ptr_ = nullptr;
}

std::string NrtModel::to_string() const {
  std::ostringstream oss;
  oss << "NrtModel(ptr=" << ptr_ << ", core_id=" << core_id_
      << ", rank_id=" << rank_id_ << ", world_size=" << world_size_
      << ", is_collective=" << (is_collective_ ? "True" : "False")
      << ", is_owner=" << (is_owner() ? "True" : "False") << ")";
  return oss.str();
}

} // namespace spike
