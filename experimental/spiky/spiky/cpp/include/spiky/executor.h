// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "spiky/model.h"
#include "spiky/device_tensor.h"

#include <cstdint>
#include <future>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" {
#include <nrt/nrt.h>
}

#include "spiky/dlpack.h"

namespace spiky {

struct RuntimeStats {
  int64_t total_executions{0};
  int64_t total_compile_count{0};
  std::map<int64_t, int64_t> bucket_hits;
  double total_pad_ratio_sum{0.0};
  double total_execution_time_ms{0.0};
  double min_execution_time_ms{1e9};
  double max_execution_time_ms{0.0};
  double total_h2d_time_ms{0.0};
  double total_nrt_exec_time_ms{0.0};
  double total_d2h_time_ms{0.0};
};

struct BundleSpec {
  std::map<int64_t, std::string> bucket_to_neff;
  std::map<int64_t, int64_t> dynamic_specs;  // arg_idx -> pad_dim
  bool cc_enabled{false};    // collective communication enabled
  uint32_t rank_id{0};       // process rank for collectives
  uint32_t world_size{1};    // total number of processes
};

class Bundle {
 public:
  Bundle() = default;
  explicit Bundle(int64_t id) : id_(id) {}

  int64_t Id() const { return id_; }

  int64_t SelectBucket(int64_t actual_len) const;
  int64_t PrimaryArgIdx() const;
  int64_t PrimaryPadDim() const;

  std::vector<int64_t> SortedBuckets() const { return sorted_bucket_sizes_; }

 private:
  friend class Engine;

  struct Bucket {
    int64_t bucket_size{0};
    std::string neff_path;
    Model model;
    bool loaded{false};
    std::vector<TensorInfo> input_infos;
    std::vector<TensorInfo> output_infos;
  };

  int64_t id_{-1};
  std::map<int64_t, Bucket> buckets_;
  std::vector<int64_t> sorted_bucket_sizes_;
  std::map<int64_t, int64_t> dynamic_specs_;
  bool cc_enabled_{false};
  uint32_t rank_id_{0};
  uint32_t world_size_{1};
};

// Global engine (bundle registry + buffer/pipeline state).
class Engine {
 public:
  static Engine& Global();

  void Init(int64_t device_id = 0);
  void Shutdown();
  bool IsInitialized() const;

  int64_t RegisterBundle(const BundleSpec& spec);
  void UnregisterBundle(int64_t bundle_id);
  int64_t SelectBucket(int64_t bundle_id, int64_t actual_len);

  // Execute with CPU host DLTensors. Returns device output tensors (nrt_tensor_t*),
  // detached from scratch so they outlive subsequent calls.
  std::vector<DeviceTensor> Execute(int64_t bundle_id, int64_t bucket_size,
                                    const std::vector<DLTensor*>& inputs,
                                    bool pad_on_device = true,
                                    bool keep_outputs_on_device = true,
                                    bool unpad_outputs = true,
                                    int64_t actual_len = 0,
                                    bool save_trace = false,
                                    const std::string& ntff_name = "");

  std::vector<DeviceTensor> ExecutePipelined(int64_t bundle_id, int64_t bucket_size,
                                             const std::vector<DLTensor*>& inputs,
                                             const std::vector<DLTensor*>& next_inputs,
                                             bool pad_on_device = true,
                                             bool keep_outputs_on_device = true,
                                             bool unpad_outputs = true,
                                             int64_t actual_len = 0,
                                             bool save_trace = false,
                                             const std::string& ntff_name = "");

  void FlushPipeline(int64_t bundle_id);

  RuntimeStats GetStats() const;
  void ResetStats();

 private:
  Engine() = default;

  Bundle& GetBundleOrThrow(int64_t bundle_id);
  const Bundle& GetBundleOrThrow(int64_t bundle_id) const;

  void EnsureBucketLoaded(Bundle& b, int64_t bucket_size);

  // Unified buffers per bundle (max-bucket scratch).
  static constexpr int NUM_BUFFERS = 2;
  struct UnifiedBuffers {
    int64_t max_bucket_size{0};
    std::vector<nrt_tensor_t*> inputs[NUM_BUFFERS];
    std::vector<nrt_tensor_t*> outputs[NUM_BUFFERS];
    std::unordered_map<int64_t, nrt_tensor_set_t*> input_sets[NUM_BUFFERS];
    std::unordered_map<int64_t, nrt_tensor_set_t*> output_sets[NUM_BUFFERS];
    std::vector<TensorInfo> input_infos;
    std::vector<TensorInfo> output_infos;
  };
  struct UnifiedPipelineState {
    std::future<void> pending_h2d;
    int active_buffer{0};
    int staging_buffer{1};
    bool first_call{true};
    int64_t prefetched_actual_len{0};
    int64_t prefetched_bucket_size{0};
  };

  void AllocateUnifiedBuffers(int64_t bundle_id, int64_t max_bucket_size,
                              const std::vector<TensorInfo>& input_infos,
                              const std::vector<TensorInfo>& output_infos);
  void CreateBucketTensorSets(int64_t bundle_id, int64_t bucket_size,
                              const std::vector<TensorInfo>& input_infos,
                              const std::vector<TensorInfo>& output_infos);
  void FreeUnifiedBuffers(int64_t bundle_id) noexcept;
  void ReallocateUnifiedBuffers(int64_t bundle_id, int64_t new_max_bucket_size,
                                const std::vector<TensorInfo>& input_infos,
                                const std::vector<TensorInfo>& output_infos);

  std::vector<DeviceTensor> DetachOutputs(int64_t bundle_id,
                                         int buffer_idx,
                                         int64_t bucket_size,
                                         int64_t actual_len,
                                         bool unpad_outputs);

  bool initialized_{false};
  int64_t device_id_{0};
  uint32_t core_id_{0};

  std::unordered_map<int64_t, Bundle> bundles_;
  int64_t next_bundle_id_{0};

  std::unordered_map<int64_t, UnifiedBuffers> unified_buffers_;
  std::unordered_map<int64_t, UnifiedPipelineState> unified_pipeline_states_;

  mutable std::mutex mutex_;
  RuntimeStats stats_;
};

}  // namespace spiky
