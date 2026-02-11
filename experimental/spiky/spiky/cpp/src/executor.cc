// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Spiky execution engine (bucketed execution + padding + unified buffers + pipelining).
//
// This implementation is derived from neuron_vm's unified-buffer path and adapted
// to use plain STL types and Spiky's shared DeviceMemoryPool.

#include "spiky/executor.h"

#include "spiky/pool.h"
#include "spiky/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

extern "C" {
#include <nrt/nrt_status.h>
#include <nrt/nrt_profile.h>
}

namespace spiky {
namespace {

inline void CheckNRT(NRT_STATUS status, const char* op) {
  if (status != NRT_SUCCESS) {
    throw std::runtime_error(std::string("spiky: ") + op + " failed, status=" +
                             std::to_string(static_cast<int>(status)));
  }
}

// dtype string -> element size
size_t ElemSizeBytes(const std::string& dtype) {
  if (dtype == "float16" || dtype == "bfloat16") return 2;
  if (dtype == "float32") return 4;
  if (dtype == "int8" || dtype == "uint8") return 1;
  if (dtype == "int16" || dtype == "uint16") return 2;
  if (dtype == "int32" || dtype == "uint32") return 4;
  if (dtype == "int64" || dtype == "uint64") return 8;
  return 4;
}

size_t CalcDLTensorBytes(const DLTensor* dl) {
  if (!dl) return 0;
  size_t num = 1;
  for (int i = 0; i < dl->ndim; ++i) num *= static_cast<size_t>(dl->shape[i]);
  size_t elem = static_cast<size_t>((dl->dtype.bits * dl->dtype.lanes + 7) / 8);
  return num * elem;
}

// Strided iteration helper for pad/unpad. "Batch" means product(dims < pad_dim).
struct StridedIterator {
  int64_t num_batches{1};
  size_t src_stride{0};
  size_t dst_stride{0};
  size_t valid_bytes{0};
  size_t trailing_bytes{0};

  static StridedIterator Create(const std::vector<int64_t>& shape,
                                int64_t pad_dim,
                                int64_t actual_len,
                                int64_t padded_len,
                                size_t elem_size) {
    StridedIterator it;
    int ndim = static_cast<int>(shape.size());
    it.num_batches = 1;
    for (int d = 0; d < pad_dim && d < ndim; ++d) it.num_batches *= shape[d];
    it.trailing_bytes = elem_size;
    for (int d = static_cast<int>(pad_dim) + 1; d < ndim; ++d) {
      it.trailing_bytes *= static_cast<size_t>(shape[d]);
    }
    it.src_stride = static_cast<size_t>(actual_len) * it.trailing_bytes;
    it.dst_stride = static_cast<size_t>(padded_len) * it.trailing_bytes;
    it.valid_bytes = static_cast<size_t>(actual_len) * it.trailing_bytes;
    return it;
  }
};

void WriteTensorToDevice(nrt_tensor_t* device_tensor, const void* host_data,
                         size_t offset, size_t size) {
  CheckNRT(nrt_tensor_write(device_tensor, host_data, offset, size), "nrt_tensor_write");
}

void ReadTensorFromDevice(nrt_tensor_t* device_tensor, void* host_data,
                          size_t offset, size_t size) {
  CheckNRT(nrt_tensor_read(device_tensor, host_data, offset, size), "nrt_tensor_read");
}

void PadTensorOnDevice(nrt_tensor_t* device_tensor,
                       size_t actual_bytes,
                       size_t expected_bytes,
                       DeviceMemoryPool& pool,
                       int device) {
  if (actual_bytes >= expected_bytes) return;
  size_t pad_bytes = expected_bytes - actual_bytes;
  nrt_tensor_t* zero = pool.GetZeroTensor(pad_bytes, device);
  CheckNRT(nrt_tensor_copy(zero, 0, device_tensor, actual_bytes, pad_bytes), "nrt_tensor_copy(pad)");
}

// Copy host->device with padding along pad_dim, preserving per-batch layout.
void CopyAndPadToDeviceStrided(nrt_tensor_t* device_tensor,
                               const DLTensor* host_tensor,
                               size_t expected_bytes,
                               int64_t original_len,
                               int64_t pad_dim,
                               DeviceMemoryPool& pool,
                               int device) {
  (void)expected_bytes;
  if (!host_tensor || !host_tensor->data) return;

  int ndim = host_tensor->ndim;
  if (ndim <= 0 || pad_dim < 0 || pad_dim >= ndim) {
    size_t actual_bytes = CalcDLTensorBytes(host_tensor);
    WriteTensorToDevice(device_tensor, host_tensor->data, 0, actual_bytes);
    return;
  }

  size_t elem_size = static_cast<size_t>((host_tensor->dtype.bits * host_tensor->dtype.lanes + 7) / 8);
  int64_t actual_pad_dim_size = host_tensor->shape[pad_dim];
  int64_t bucket_pad_dim_size = original_len;  // expected padded len for this bucket

  // If no padding needed, just write full data.
  if (actual_pad_dim_size >= bucket_pad_dim_size) {
    size_t actual_bytes = CalcDLTensorBytes(host_tensor);
    WriteTensorToDevice(device_tensor, host_tensor->data, 0, actual_bytes);
    return;
  }

  // Leading dims product.
  int64_t leading = 1;
  for (int d = 0; d < pad_dim; ++d) leading *= host_tensor->shape[d];

  // Trailing bytes (dims after pad_dim).
  size_t trailing_bytes = elem_size;
  for (int d = static_cast<int>(pad_dim) + 1; d < ndim; ++d) {
    trailing_bytes *= static_cast<size_t>(host_tensor->shape[d]);
  }

  size_t actual_slice_bytes = static_cast<size_t>(actual_pad_dim_size) * trailing_bytes;
  size_t bucket_slice_bytes = static_cast<size_t>(bucket_pad_dim_size) * trailing_bytes;
  size_t pad_bytes_per_slice = static_cast<size_t>(bucket_pad_dim_size - actual_pad_dim_size) * trailing_bytes;

  nrt_tensor_t* zero_tensor = pool.GetZeroTensor(pad_bytes_per_slice, device);
  const char* host_data = static_cast<const char*>(host_tensor->data);

  for (int64_t idx = 0; idx < leading; ++idx) {
    size_t src_off = static_cast<size_t>(idx) * actual_slice_bytes;
    size_t dst_off = static_cast<size_t>(idx) * bucket_slice_bytes;
    WriteTensorToDevice(device_tensor, host_data + src_off, dst_off, actual_slice_bytes);
    CheckNRT(nrt_tensor_copy(zero_tensor, 0, device_tensor, dst_off + actual_slice_bytes, pad_bytes_per_slice),
             "nrt_tensor_copy(strided_pad)");
  }
}

// Copy host->device into unified max-bucket layout (pad_dim stride uses max_bucket_size).
void CopyToDeviceWithMaxBucketStride(nrt_tensor_t* device_tensor,
                                     const DLTensor* host_tensor,
                                     int64_t max_bucket_size,
                                     int64_t pad_dim,
                                     DeviceMemoryPool& pool,
                                     int device,
                                     int64_t actual_len,
                                     int64_t bucket_len) {
  if (!host_tensor || !host_tensor->data) return;
  int ndim = host_tensor->ndim;
  size_t elem_size = static_cast<size_t>((host_tensor->dtype.bits * host_tensor->dtype.lanes + 7) / 8);

  // Fall back to simple copy/pad if invalid pad_dim.
  if (ndim <= 0 || pad_dim < 0 || pad_dim >= ndim) {
    size_t actual_bytes = CalcDLTensorBytes(host_tensor);
    WriteTensorToDevice(device_tensor, host_tensor->data, 0, actual_bytes);
    return;
  }

  // Leading dims product.
  int64_t leading = 1;
  for (int d = 0; d < pad_dim; ++d) leading *= host_tensor->shape[d];

  // Trailing bytes (dims after pad_dim).
  size_t trailing_bytes = elem_size;
  for (int d = static_cast<int>(pad_dim) + 1; d < ndim; ++d) {
    trailing_bytes *= static_cast<size_t>(host_tensor->shape[d]);
  }

  int64_t actual_pad_dim_size = host_tensor->shape[pad_dim];
  int64_t actual_use = (actual_len > 0) ? actual_len : actual_pad_dim_size;
  int64_t bucket_use = (bucket_len > 0) ? bucket_len : actual_use;

  size_t actual_slice_bytes = static_cast<size_t>(actual_use) * trailing_bytes;
  size_t bucket_slice_bytes = static_cast<size_t>(bucket_use) * trailing_bytes;
  size_t max_slice_bytes = static_cast<size_t>(max_bucket_size) * trailing_bytes;
  size_t pad_bytes_per_slice = (bucket_use > actual_use)
                                   ? static_cast<size_t>(bucket_use - actual_use) * trailing_bytes
                                   : 0;

  nrt_tensor_t* zero_tensor = (pad_bytes_per_slice > 0) ? pool.GetZeroTensor(pad_bytes_per_slice, device) : nullptr;
  const char* host_data = static_cast<const char*>(host_tensor->data);

  for (int64_t idx = 0; idx < leading; ++idx) {
    size_t src_off = static_cast<size_t>(idx) * actual_slice_bytes;
    size_t dst_off = static_cast<size_t>(idx) * max_slice_bytes;

    // Copy actual data.
    if (actual_slice_bytes > 0) {
      WriteTensorToDevice(device_tensor, host_data + src_off, dst_off, actual_slice_bytes);
    }
    // Pad zeros up to bucket_len (within the max stride region).
    if (pad_bytes_per_slice > 0 && zero_tensor) {
      CheckNRT(nrt_tensor_copy(zero_tensor, 0, device_tensor, dst_off + actual_slice_bytes, pad_bytes_per_slice),
               "nrt_tensor_copy(max_stride_pad)");
    }
    // Note: bytes beyond bucket_len inside the max_slice_bytes region are left untouched;
    // the executed model should only read up to its bucket_len.
    (void)bucket_slice_bytes;
  }
}

void UnpadFromDeviceStrided(nrt_tensor_t* src_tensor, nrt_tensor_t* dst_tensor,
                            const std::vector<int64_t>& padded_shape,
                            int64_t original_len, int64_t pad_dim, size_t elem_size) {
  int ndim = static_cast<int>(padded_shape.size());
  int64_t padded_dim_size = padded_shape[static_cast<size_t>(pad_dim)];
  if (original_len >= padded_dim_size) {
    size_t total_bytes = elem_size;
    for (int d = 0; d < ndim; ++d) total_bytes *= static_cast<size_t>(padded_shape[d]);
    CheckNRT(nrt_tensor_copy(src_tensor, 0, dst_tensor, 0, total_bytes), "nrt_tensor_copy(unpad_full)");
    return;
  }

  auto it = StridedIterator::Create(padded_shape, pad_dim, original_len, padded_dim_size, elem_size);
  for (int64_t b = 0; b < it.num_batches; ++b) {
    size_t src_off = static_cast<size_t>(b) * it.dst_stride;
    size_t dst_off = static_cast<size_t>(b) * it.src_stride;
    CheckNRT(nrt_tensor_copy(src_tensor, src_off, dst_tensor, dst_off, it.valid_bytes), "nrt_tensor_copy(unpad_strided)");
  }
}

void UnpadFromDeviceWithMaxBucketStride(nrt_tensor_t* src_tensor, nrt_tensor_t* dst_tensor,
                                       int64_t max_bucket_size, int64_t original_len,
                                       int64_t pad_dim, const std::vector<int64_t>& output_shape,
                                       size_t elem_size) {
  int ndim = static_cast<int>(output_shape.size());
  if (ndim < 2 || pad_dim < 0 || pad_dim >= ndim) {
    size_t total = elem_size;
    for (int d = 0; d < ndim; ++d) total *= static_cast<size_t>(output_shape[d]);
    CheckNRT(nrt_tensor_copy(src_tensor, 0, dst_tensor, 0, total), "nrt_tensor_copy(unpad_full_max)");
    return;
  }

  int64_t inner = 1;
  for (int d = static_cast<int>(pad_dim) + 1; d < ndim; ++d) inner *= output_shape[d];
  int64_t max_stride = max_bucket_size * inner * static_cast<int64_t>(elem_size);
  int64_t out_stride = original_len * inner * static_cast<int64_t>(elem_size);

  int64_t num_batches = 1;
  for (int d = 0; d < pad_dim; ++d) num_batches *= output_shape[d];

  for (int64_t b = 0; b < num_batches; ++b) {
    size_t src_off = static_cast<size_t>(b * max_stride);
    size_t dst_off = static_cast<size_t>(b * out_stride);
    CheckNRT(nrt_tensor_copy(src_tensor, src_off, dst_tensor, dst_off, static_cast<size_t>(out_stride)),
             "nrt_tensor_copy(unpad_max_stride)");
  }
}

int64_t DetectActualLen(const std::vector<DLTensor*>& inputs,
                        const std::map<int64_t, int64_t>& dynamic_specs) {
  if (dynamic_specs.empty()) return 0;
  int64_t primary_arg = dynamic_specs.begin()->first;
  int64_t pad_dim = dynamic_specs.begin()->second;
  if (primary_arg < 0 || static_cast<size_t>(primary_arg) >= inputs.size()) return 0;
  const DLTensor* t = inputs[static_cast<size_t>(primary_arg)];
  if (!t) return 0;
  if (pad_dim < 0 || pad_dim >= t->ndim) return 0;
  return t->shape[pad_dim];
}

}  // namespace

int64_t Bundle::PrimaryArgIdx() const {
  if (dynamic_specs_.empty()) return -1;
  return dynamic_specs_.begin()->first;
}

int64_t Bundle::PrimaryPadDim() const {
  if (dynamic_specs_.empty()) return -1;
  return dynamic_specs_.begin()->second;
}

int64_t Bundle::SelectBucket(int64_t actual_len) const {
  for (int64_t b : sorted_bucket_sizes_) {
    if (b >= actual_len) return b;
  }
  return sorted_bucket_sizes_.empty() ? -1 : sorted_bucket_sizes_.back();
}

Engine& Engine::Global() {
  static Engine e;
  return e;
}

void Engine::Init(int64_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) return;
  device_id_ = device_id;
  core_id_ = static_cast<uint32_t>(device_id);
  initialized_ = true;
}

void Engine::Shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& kv : unified_buffers_) {
    FreeUnifiedBuffers(kv.first);
  }
  unified_buffers_.clear();
  unified_pipeline_states_.clear();
  bundles_.clear();
  initialized_ = false;
}

bool Engine::IsInitialized() const { return initialized_; }

Bundle& Engine::GetBundleOrThrow(int64_t bundle_id) {
  auto it = bundles_.find(bundle_id);
  if (it == bundles_.end()) throw std::runtime_error("spiky: unknown bundle_id");
  return it->second;
}

const Bundle& Engine::GetBundleOrThrow(int64_t bundle_id) const {
  auto it = bundles_.find(bundle_id);
  if (it == bundles_.end()) throw std::runtime_error("spiky: unknown bundle_id");
  return it->second;
}

int64_t Engine::RegisterBundle(const BundleSpec& spec) {
  std::lock_guard<std::mutex> lock(mutex_);
  int64_t id = next_bundle_id_++;
  Bundle b(id);
  b.dynamic_specs_ = spec.dynamic_specs;
  b.cc_enabled_ = spec.cc_enabled;
  b.rank_id_ = spec.rank_id;
  b.world_size_ = spec.world_size;

  for (const auto& kv : spec.bucket_to_neff) {
    Bundle::Bucket bk;
    bk.bucket_size = kv.first;
    bk.neff_path = kv.second;
    b.buckets_[kv.first] = std::move(bk);
    b.sorted_bucket_sizes_.push_back(kv.first);
  }
  std::sort(b.sorted_bucket_sizes_.begin(), b.sorted_bucket_sizes_.end());
  bundles_.emplace(id, std::move(b));
  return id;
}

void Engine::UnregisterBundle(int64_t bundle_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  FreeUnifiedBuffers(bundle_id);
  unified_buffers_.erase(bundle_id);
  unified_pipeline_states_.erase(bundle_id);
  bundles_.erase(bundle_id);
}

int64_t Engine::SelectBucket(int64_t bundle_id, int64_t actual_len) {
  std::lock_guard<std::mutex> lock(mutex_);
  const Bundle& b = GetBundleOrThrow(bundle_id);
  return b.SelectBucket(actual_len);
}

void Engine::EnsureBucketLoaded(Bundle& b, int64_t bucket_size) {
  auto it = b.buckets_.find(bucket_size);
  if (it == b.buckets_.end()) throw std::runtime_error("spiky: unknown bucket_size");
  auto& bk = it->second;
  if (bk.loaded) return;
  bk.model.LoadFromFile(bk.neff_path, core_id_, b.cc_enabled_, b.rank_id_, b.world_size_);
  auto [ins, outs] = bk.model.GetTensorInfo();
  bk.input_infos = std::move(ins);
  bk.output_infos = std::move(outs);
  bk.loaded = true;
}

void Engine::AllocateUnifiedBuffers(int64_t bundle_id, int64_t max_bucket_size,
                                    const std::vector<TensorInfo>& input_infos,
                                    const std::vector<TensorInfo>& output_infos) {
  UnifiedBuffers ub;
  ub.max_bucket_size = max_bucket_size;
  ub.input_infos = input_infos;
  ub.output_infos = output_infos;

  DeviceMemoryPool& pool = DeviceMemoryPool::Global();
  int dev = static_cast<int>(core_id_);

  for (int buf = 0; buf < NUM_BUFFERS; ++buf) {
    const char* suffix = (buf == 0) ? "A" : "B";
    ub.inputs[buf].reserve(input_infos.size());
    ub.outputs[buf].reserve(output_infos.size());
    for (size_t i = 0; i < input_infos.size(); ++i) {
      ub.inputs[buf].push_back(pool.Acquire(static_cast<size_t>(input_infos[i].size_bytes), dev, suffix));
    }
    for (size_t i = 0; i < output_infos.size(); ++i) {
      ub.outputs[buf].push_back(pool.Acquire(static_cast<size_t>(output_infos[i].size_bytes), dev, suffix));
    }
  }

  unified_buffers_[bundle_id] = std::move(ub);
  unified_pipeline_states_[bundle_id] = UnifiedPipelineState{};
}

void Engine::CreateBucketTensorSets(int64_t bundle_id, int64_t bucket_size,
                                    const std::vector<TensorInfo>& input_infos,
                                    const std::vector<TensorInfo>& output_infos) {
  auto ub_it = unified_buffers_.find(bundle_id);
  if (ub_it == unified_buffers_.end()) {
    throw std::runtime_error("spiky: unified buffers missing");
  }
  auto& ub = ub_it->second;

  for (int buf = 0; buf < NUM_BUFFERS; ++buf) {
    if (ub.input_sets[buf].count(bucket_size)) continue;

    nrt_tensor_set_t* iset = nullptr;
    CheckNRT(nrt_allocate_tensor_set(&iset), "nrt_allocate_tensor_set(input)");
    for (size_t i = 0; i < input_infos.size(); ++i) {
      CheckNRT(nrt_add_tensor_to_tensor_set(iset, input_infos[i].name.c_str(), ub.inputs[buf][i]),
               "nrt_add_tensor_to_tensor_set(input)");
    }
    ub.input_sets[buf][bucket_size] = iset;

    nrt_tensor_set_t* oset = nullptr;
    CheckNRT(nrt_allocate_tensor_set(&oset), "nrt_allocate_tensor_set(output)");
    for (size_t i = 0; i < output_infos.size(); ++i) {
      CheckNRT(nrt_add_tensor_to_tensor_set(oset, output_infos[i].name.c_str(), ub.outputs[buf][i]),
               "nrt_add_tensor_to_tensor_set(output)");
    }
    ub.output_sets[buf][bucket_size] = oset;
  }
}

void Engine::FreeUnifiedBuffers(int64_t bundle_id) noexcept {
  auto it = unified_buffers_.find(bundle_id);
  if (it == unified_buffers_.end()) return;
  UnifiedBuffers& ub = it->second;

  // Wait pending async H2D.
  auto st_it = unified_pipeline_states_.find(bundle_id);
  if (st_it != unified_pipeline_states_.end()) {
    if (st_it->second.pending_h2d.valid()) {
      try {
        st_it->second.pending_h2d.get();
      } catch (...) {
      }
    }
    unified_pipeline_states_.erase(st_it);
  }

  DeviceMemoryPool& pool = DeviceMemoryPool::Global();

  // Suppress NRT "Unexpected runtime state: NRT_STATE_START" warnings from
  // nrt_destroy_tensor_set.  The Engine loads NEFFs directly so the NRT state
  // machine never advances past START, but tensor sets are still freed.
  int saved_stderr = dup(STDERR_FILENO);
  FILE* devnull = fopen("/dev/null", "w");
  if (devnull) {
    fflush(stderr);
    dup2(fileno(devnull), STDERR_FILENO);
    fclose(devnull);
  }

  for (int buf = 0; buf < NUM_BUFFERS; ++buf) {
    for (auto& kv : ub.input_sets[buf]) {
      if (kv.second) nrt_destroy_tensor_set(&kv.second);
    }
    for (auto& kv : ub.output_sets[buf]) {
      if (kv.second) nrt_destroy_tensor_set(&kv.second);
    }
    ub.input_sets[buf].clear();
    ub.output_sets[buf].clear();

    for (auto* t : ub.inputs[buf]) pool.Release(t);
    for (auto* t : ub.outputs[buf]) pool.Release(t);
    ub.inputs[buf].clear();
    ub.outputs[buf].clear();
  }

  // Restore stderr.
  if (saved_stderr >= 0) {
    fflush(stderr);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
  }
}

void Engine::ReallocateUnifiedBuffers(int64_t bundle_id, int64_t new_max_bucket_size,
                                      const std::vector<TensorInfo>& input_infos,
                                      const std::vector<TensorInfo>& output_infos) {
  FreeUnifiedBuffers(bundle_id);
  unified_buffers_.erase(bundle_id);
  AllocateUnifiedBuffers(bundle_id, new_max_bucket_size, input_infos, output_infos);
}

// Copy current inputs into a given unified input buffer.
void SetupInputTensors(const std::vector<DLTensor*>& inputs,
                       const std::vector<nrt_tensor_t*>& dst_device_tensors,
                       const std::vector<TensorInfo>& input_infos,
                       bool pad_on_device,
                       DeviceMemoryPool& pool,
                       int device,
                       const std::map<int64_t, int64_t>& dynamic_specs,
                       int64_t max_bucket_size,
                       int64_t bucket_size,
                       int64_t effective_actual_len) {
  // For each input, copy bytes and optionally pad.
  for (size_t i = 0; i < dst_device_tensors.size() && i < input_infos.size(); ++i) {
    const DLTensor* dl = (i < inputs.size()) ? inputs[i] : nullptr;
    nrt_tensor_t* dev_tensor = dst_device_tensors[i];
    if (!dl || !dl->data || !dev_tensor) continue;

    size_t actual_bytes = CalcDLTensorBytes(dl);
    size_t expected_bytes = static_cast<size_t>(input_infos[i].size_bytes);
    size_t copy_bytes = std::min(actual_bytes, expected_bytes);

    bool is_dynamic = dynamic_specs.count(static_cast<int64_t>(i)) != 0;
    if (pad_on_device && is_dynamic) {
      int64_t pad_dim = dynamic_specs.at(static_cast<int64_t>(i));
      int64_t actual_len = (effective_actual_len > 0) ? effective_actual_len
                                                      : (pad_dim >= 0 && pad_dim < dl->ndim ? dl->shape[pad_dim] : 0);
      if (max_bucket_size > 0) {
        CopyToDeviceWithMaxBucketStride(dev_tensor, dl, max_bucket_size, pad_dim,
                                        pool, device, actual_len, bucket_size);
      } else {
        CopyAndPadToDeviceStrided(dev_tensor, dl, expected_bytes, bucket_size, pad_dim, pool, device);
      }
    } else {
      if (copy_bytes > 0) WriteTensorToDevice(dev_tensor, dl->data, 0, copy_bytes);
      if (pad_on_device) PadTensorOnDevice(dev_tensor, copy_bytes, expected_bytes, pool, device);
    }
  }
}

std::future<void> StartAsyncH2D(const std::vector<DLTensor*>& inputs,
                                std::vector<nrt_tensor_t*>* dst,
                                const std::vector<TensorInfo>& input_infos,
                                bool pad_on_device,
                                DeviceMemoryPool& pool,
                                int device,
                                const std::map<int64_t, int64_t>& dynamic_specs,
                                int64_t max_bucket_size,
                                int64_t bucket_size,
                                int64_t effective_actual_len) {
  // Deep-copy DLTensor metadata so the async thread doesn't reference
  // caller-owned temporaries (e.g. DLTensorHolders in the Python binding).
  // DLTensor::data still points to the original numpy buffer, which Python
  // keeps alive via reference counting.
  auto owned_dls = std::make_shared<std::vector<DLTensor>>();
  auto owned_shapes = std::make_shared<std::vector<std::vector<int64_t>>>();
  owned_dls->reserve(inputs.size());
  owned_shapes->reserve(inputs.size());
  for (const DLTensor* dl : inputs) {
    owned_shapes->emplace_back(dl->shape, dl->shape + dl->ndim);
    DLTensor copy = *dl;
    copy.shape = owned_shapes->back().data();
    copy.strides = nullptr;
    owned_dls->push_back(copy);
  }
  auto owned_ptrs = std::make_shared<std::vector<DLTensor*>>();
  owned_ptrs->reserve(owned_dls->size());
  for (auto& dl : *owned_dls) owned_ptrs->push_back(&dl);

  return std::async(std::launch::async,
      [owned_dls, owned_shapes, owned_ptrs, dst, &input_infos, pad_on_device,
       &pool, device, &dynamic_specs, max_bucket_size, bucket_size,
       effective_actual_len]() {
    SetupInputTensors(*owned_ptrs, *dst, input_infos, pad_on_device, pool, device,
                      dynamic_specs, max_bucket_size, bucket_size, effective_actual_len);
  });
}

std::vector<DeviceTensor> Engine::DetachOutputs(int64_t bundle_id, int buffer_idx,
                                               int64_t bucket_size, int64_t actual_len,
                                               bool unpad_outputs) {
  auto ub_it = unified_buffers_.find(bundle_id);
  if (ub_it == unified_buffers_.end()) return {};
  UnifiedBuffers& ub = ub_it->second;

  DeviceMemoryPool& pool = DeviceMemoryPool::Global();
  int dev = static_cast<int>(core_id_);

  // Use primary pad_dim for unpadding.
  const Bundle& bundle = GetBundleOrThrow(bundle_id);
  int64_t pad_dim = bundle.PrimaryPadDim();

  std::vector<DeviceTensor> outputs;
  outputs.reserve(ub.outputs[buffer_idx].size());

  for (size_t i = 0; i < ub.outputs[buffer_idx].size() && i < ub.output_infos.size(); ++i) {
    nrt_tensor_t* scratch = ub.outputs[buffer_idx][i];
    const TensorInfo& info = ub.output_infos[i];
    if (!scratch) continue;

    std::vector<int64_t> padded_shape = info.shape;
    size_t elem_size = ElemSizeBytes(info.dtype);

    // Only unpad outputs whose pad_dim size matches a bucket size (was actually padded).
    // Reduction outputs (e.g., sum(dim=0)) have different sizes and must be skipped.
    bool needs_unpad = unpad_outputs && (actual_len > 0) &&
                       (pad_dim >= 0) &&
                       (pad_dim < static_cast<int64_t>(padded_shape.size())) &&
                       (actual_len < padded_shape[static_cast<size_t>(pad_dim)]) &&
                       (padded_shape[static_cast<size_t>(pad_dim)] == bucket_size ||
                        padded_shape[static_cast<size_t>(pad_dim)] == ub.max_bucket_size);

    std::vector<int64_t> out_shape = padded_shape;
    if (needs_unpad) out_shape[static_cast<size_t>(pad_dim)] = actual_len;

    size_t out_bytes = elem_size;
    for (int64_t d : out_shape) out_bytes *= static_cast<size_t>(d);

    nrt_tensor_t* detached = pool.Acquire(out_bytes, dev, "spiky_detached_out");
    if (needs_unpad) {
      if (ub.max_bucket_size > 0) {
        UnpadFromDeviceWithMaxBucketStride(scratch, detached, ub.max_bucket_size,
                                           actual_len, pad_dim, out_shape, elem_size);
      } else {
        UnpadFromDeviceStrided(scratch, detached, padded_shape, actual_len, pad_dim, elem_size);
      }
    } else {
      CheckNRT(nrt_tensor_copy(scratch, 0, detached, 0, out_bytes), "nrt_tensor_copy(detach)");
    }

    outputs.emplace_back(detached, dev, out_bytes, out_shape, info.dtype);
  }

  return outputs;
}

std::vector<DeviceTensor> Engine::Execute(int64_t bundle_id, int64_t bucket_size,
                                         const std::vector<DLTensor*>& inputs,
                                         bool pad_on_device,
                                         bool keep_outputs_on_device,
                                         bool unpad_outputs,
                                         int64_t actual_len,
                                         bool save_trace,
                                         const std::string& ntff_name) {
  (void)keep_outputs_on_device;
  auto start = std::chrono::high_resolution_clock::now();

  if (!Runtime::Global().IsInitialized()) {
    throw std::runtime_error("spiky: runtime not initialized (call spiky.init())");
  }

  // Phase 1: Registry setup under global lock.
  Bundle* bundle_ptr;
  Bundle::Bucket* bk_ptr;
  UnifiedBuffers* ub_ptr;
  bool is_static;
  int64_t effective_actual_len;
  int64_t effective_bucket;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    bundle_ptr = &GetBundleOrThrow(bundle_id);
    Bundle& bundle = *bundle_ptr;
    EnsureBucketLoaded(bundle, bucket_size);
    bk_ptr = &bundle.buckets_.at(bucket_size);

    is_static = bundle.dynamic_specs_.empty()
                && bundle.sorted_bucket_sizes_.size() == 1;

    effective_actual_len = is_static ? 0
        : ((actual_len > 0) ? actual_len : DetectActualLen(inputs, bundle.dynamic_specs_));
    effective_bucket = bucket_size;

    int64_t max_bucket = bundle.sorted_bucket_sizes_.empty() ? bucket_size : bundle.sorted_bucket_sizes_.back();
    if (max_bucket != bucket_size) {
      EnsureBucketLoaded(bundle, max_bucket);
    }

    if (unified_buffers_.count(bundle_id) == 0) {
      auto& max_bk = bundle.buckets_.at(max_bucket);
      AllocateUnifiedBuffers(bundle_id, max_bucket, max_bk.input_infos, max_bk.output_infos);
    }

    ub_ptr = &unified_buffers_.at(bundle_id);
    if (ub_ptr->max_bucket_size < max_bucket) {
      auto& max_bk = bundle.buckets_.at(max_bucket);
      ReallocateUnifiedBuffers(bundle_id, max_bucket, max_bk.input_infos, max_bk.output_infos);
      ub_ptr = &unified_buffers_.at(bundle_id);
    }

    CreateBucketTensorSets(bundle_id, bucket_size, bk_ptr->input_infos, bk_ptr->output_infos);
  }

  // Phase 2: Execution under per-bundle lock.
  std::lock_guard<std::mutex> bundle_lock(bundle_ptr->Mutex());
  Bundle& bundle = *bundle_ptr;
  auto& bk = *bk_ptr;
  UnifiedBuffers& ub = *ub_ptr;

  int buf_idx = 0;
  DeviceMemoryPool& pool = DeviceMemoryPool::Global();
  int dev = static_cast<int>(core_id_);

  auto t_h2d_start = std::chrono::high_resolution_clock::now();
  SetupInputTensors(inputs, ub.inputs[buf_idx], ub.input_infos, pad_on_device, pool, dev,
                    bundle.dynamic_specs_, ub.max_bucket_size, effective_bucket, effective_actual_len);
  auto t_h2d_end = std::chrono::high_resolution_clock::now();

  auto t_exec_start = std::chrono::high_resolution_clock::now();
  if (save_trace && !ntff_name.empty()) {
    nrt_profile_start(bk.model.Ptr(), ntff_name.c_str());
  }
  CheckNRT(nrt_execute(bk.model.Ptr(), ub.input_sets[buf_idx][bucket_size], ub.output_sets[buf_idx][bucket_size]),
           "nrt_execute");
  if (save_trace && !ntff_name.empty()) {
    nrt_profile_stop(ntff_name.c_str());
  }
  auto t_exec_end = std::chrono::high_resolution_clock::now();

  auto t_detach_start = std::chrono::high_resolution_clock::now();
  std::vector<DeviceTensor> outputs = DetachOutputs(bundle_id, buf_idx, bucket_size, effective_actual_len, unpad_outputs);
  auto t_detach_end = std::chrono::high_resolution_clock::now();

  auto end = std::chrono::high_resolution_clock::now();

  // Phase 3: Stats update under global lock.
  double h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();
  double exec_ms = std::chrono::duration<double, std::milli>(t_exec_end - t_exec_start).count();
  double d2d_ms = std::chrono::duration<double, std::milli>(t_detach_end - t_detach_start).count();
  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.total_executions++;
    stats_.bucket_hits[bucket_size]++;
    stats_.total_execution_time_ms += total_ms;
    stats_.total_h2d_time_ms += h2d_ms;
    stats_.total_nrt_exec_time_ms += exec_ms;
    stats_.total_d2h_time_ms += d2d_ms;
    stats_.min_execution_time_ms = std::min(stats_.min_execution_time_ms, total_ms);
    stats_.max_execution_time_ms = std::max(stats_.max_execution_time_ms, total_ms);
    if (effective_actual_len > 0) {
      stats_.total_pad_ratio_sum += static_cast<double>(bucket_size) / static_cast<double>(effective_actual_len);
    }
  }

  return outputs;
}

std::vector<DeviceTensor> Engine::ExecutePipelined(int64_t bundle_id, int64_t bucket_size,
                                                  const std::vector<DLTensor*>& inputs,
                                                  const std::vector<DLTensor*>& next_inputs,
                                                  bool pad_on_device,
                                                  bool keep_outputs_on_device,
                                                  bool unpad_outputs,
                                                  int64_t actual_len,
                                                  bool save_trace,
                                                  const std::string& ntff_name) {
  (void)keep_outputs_on_device;
  auto start = std::chrono::high_resolution_clock::now();

  if (!Runtime::Global().IsInitialized()) {
    throw std::runtime_error("spiky: runtime not initialized (call spiky.init())");
  }

  // Phase 1: Registry setup under global lock.
  Bundle* bundle_ptr;
  Bundle::Bucket* bk_ptr;
  UnifiedBuffers* ub_ptr;
  UnifiedPipelineState* st_ptr;
  int64_t effective_actual_len;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    bundle_ptr = &GetBundleOrThrow(bundle_id);
    Bundle& bundle = *bundle_ptr;
    EnsureBucketLoaded(bundle, bucket_size);
    bk_ptr = &bundle.buckets_.at(bucket_size);

    const bool is_static = bundle.dynamic_specs_.empty()
                         && bundle.sorted_bucket_sizes_.size() == 1;

    effective_actual_len = is_static ? 0
        : ((actual_len > 0) ? actual_len : DetectActualLen(inputs, bundle.dynamic_specs_));

    int64_t max_bucket = bundle.sorted_bucket_sizes_.empty() ? bucket_size : bundle.sorted_bucket_sizes_.back();
    if (max_bucket != bucket_size) {
      EnsureBucketLoaded(bundle, max_bucket);
    }

    if (unified_buffers_.count(bundle_id) == 0) {
      auto& max_bk = bundle.buckets_.at(max_bucket);
      AllocateUnifiedBuffers(bundle_id, max_bucket, max_bk.input_infos, max_bk.output_infos);
    }

    ub_ptr = &unified_buffers_.at(bundle_id);
    if (ub_ptr->max_bucket_size < max_bucket) {
      auto& max_bk = bundle.buckets_.at(max_bucket);
      ReallocateUnifiedBuffers(bundle_id, max_bucket, max_bk.input_infos, max_bk.output_infos);
      ub_ptr = &unified_buffers_.at(bundle_id);
    }
    CreateBucketTensorSets(bundle_id, bucket_size, bk_ptr->input_infos, bk_ptr->output_infos);

    st_ptr = &unified_pipeline_states_.at(bundle_id);
  }

  // Phase 2: Execution under per-bundle lock.
  std::lock_guard<std::mutex> bundle_lock(bundle_ptr->Mutex());
  Bundle& bundle = *bundle_ptr;
  auto& bk = *bk_ptr;
  UnifiedBuffers& ub = *ub_ptr;
  UnifiedPipelineState& st = *st_ptr;
  DeviceMemoryPool& pool = DeviceMemoryPool::Global();
  int dev = static_cast<int>(core_id_);

  auto t_h2d_start = std::chrono::high_resolution_clock::now();

  if (st.first_call) {
    // Sync H2D, execute, then prefetch next.
    SetupInputTensors(inputs, ub.inputs[st.active_buffer], ub.input_infos, pad_on_device,
                      pool, dev, bundle.dynamic_specs_, ub.max_bucket_size, bucket_size, effective_actual_len);
    auto t_h2d_end = std::chrono::high_resolution_clock::now();

    auto t_exec_start = std::chrono::high_resolution_clock::now();
    if (save_trace && !ntff_name.empty()) {
      nrt_profile_start(bk.model.Ptr(), ntff_name.c_str());
    }
    CheckNRT(nrt_execute(bk.model.Ptr(),
                         ub.input_sets[st.active_buffer][bucket_size],
                         ub.output_sets[st.active_buffer][bucket_size]),
             "nrt_execute(pipelined:first)");
    if (save_trace && !ntff_name.empty()) {
      nrt_profile_stop(ntff_name.c_str());
    }
    auto t_exec_end = std::chrono::high_resolution_clock::now();

    auto t_detach_start = std::chrono::high_resolution_clock::now();
    auto outputs = DetachOutputs(bundle_id, st.active_buffer, bucket_size, effective_actual_len, unpad_outputs);
    auto t_detach_end = std::chrono::high_resolution_clock::now();

    if (!next_inputs.empty()) {
      int64_t next_actual = DetectActualLen(next_inputs, bundle.dynamic_specs_);
      int64_t next_bucket = bundle.SelectBucket(next_actual);
      st.pending_h2d = StartAsyncH2D(next_inputs, &ub.inputs[st.staging_buffer],
                                    ub.input_infos, pad_on_device, pool, dev,
                                    bundle.dynamic_specs_, ub.max_bucket_size,
                                    next_bucket, next_actual);
      st.prefetched_actual_len = next_actual;
      st.prefetched_bucket_size = next_bucket;
      std::swap(st.active_buffer, st.staging_buffer);
    }

    st.first_call = false;

    {
      double h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();
      double exec_ms = std::chrono::duration<double, std::milli>(t_exec_end - t_exec_start).count();
      double d2d_ms = std::chrono::duration<double, std::milli>(t_detach_end - t_detach_start).count();
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.total_h2d_time_ms += h2d_ms;
      stats_.total_nrt_exec_time_ms += exec_ms;
      stats_.total_d2h_time_ms += d2d_ms;
      stats_.total_executions++;
      stats_.bucket_hits[bucket_size]++;
      stats_.total_execution_time_ms += std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - start).count();
    }
    return outputs;
  }

  // Prefetch path
  bool prefetch_hit = st.pending_h2d.valid() &&
                      st.prefetched_bucket_size == bucket_size &&
                      st.prefetched_actual_len == effective_actual_len;

  if (prefetch_hit) {
    st.pending_h2d.get();  // wait for async H2D into active buffer
    auto t_h2d_end = std::chrono::high_resolution_clock::now();

    auto t_exec_start = std::chrono::high_resolution_clock::now();
    if (save_trace && !ntff_name.empty()) {
      nrt_profile_start(bk.model.Ptr(), ntff_name.c_str());
    }
    CheckNRT(nrt_execute(bk.model.Ptr(),
                         ub.input_sets[st.active_buffer][bucket_size],
                         ub.output_sets[st.active_buffer][bucket_size]),
             "nrt_execute(pipelined:hit)");
    if (save_trace && !ntff_name.empty()) {
      nrt_profile_stop(ntff_name.c_str());
    }
    auto t_exec_end = std::chrono::high_resolution_clock::now();

    auto t_detach_start = std::chrono::high_resolution_clock::now();
    auto outputs = DetachOutputs(bundle_id, st.active_buffer, bucket_size, effective_actual_len, unpad_outputs);
    auto t_detach_end = std::chrono::high_resolution_clock::now();

    if (!next_inputs.empty()) {
      int64_t next_actual = DetectActualLen(next_inputs, bundle.dynamic_specs_);
      int64_t next_bucket = bundle.SelectBucket(next_actual);
      st.pending_h2d = StartAsyncH2D(next_inputs, &ub.inputs[st.staging_buffer],
                                    ub.input_infos, pad_on_device, pool, dev,
                                    bundle.dynamic_specs_, ub.max_bucket_size,
                                    next_bucket, next_actual);
      st.prefetched_actual_len = next_actual;
      st.prefetched_bucket_size = next_bucket;
      std::swap(st.active_buffer, st.staging_buffer);
    } else {
      st.prefetched_actual_len = 0;
      st.prefetched_bucket_size = 0;
    }

    {
      double h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();
      double exec_ms = std::chrono::duration<double, std::milli>(t_exec_end - t_exec_start).count();
      double d2d_ms = std::chrono::duration<double, std::milli>(t_detach_end - t_detach_start).count();
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.total_h2d_time_ms += h2d_ms;
      stats_.total_nrt_exec_time_ms += exec_ms;
      stats_.total_d2h_time_ms += d2d_ms;
      stats_.total_executions++;
      stats_.bucket_hits[bucket_size]++;
      stats_.total_execution_time_ms += std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - start).count();
    }
    return outputs;
  }

  // Prefetch miss: wait/discard pending and do sync H2D.
  if (st.pending_h2d.valid()) {
    st.pending_h2d.get();
  }

  SetupInputTensors(inputs, ub.inputs[st.active_buffer], ub.input_infos, pad_on_device, pool, dev,
                    bundle.dynamic_specs_, ub.max_bucket_size, bucket_size, effective_actual_len);
  auto t_h2d_end = std::chrono::high_resolution_clock::now();

  auto t_exec_start = std::chrono::high_resolution_clock::now();
  if (save_trace && !ntff_name.empty()) {
    nrt_profile_start(bk.model.Ptr(), ntff_name.c_str());
  }
  CheckNRT(nrt_execute(bk.model.Ptr(),
                       ub.input_sets[st.active_buffer][bucket_size],
                       ub.output_sets[st.active_buffer][bucket_size]),
           "nrt_execute(pipelined:miss)");
  if (save_trace && !ntff_name.empty()) {
    nrt_profile_stop(ntff_name.c_str());
  }
  auto t_exec_end = std::chrono::high_resolution_clock::now();

  auto t_detach_start = std::chrono::high_resolution_clock::now();
  auto outputs = DetachOutputs(bundle_id, st.active_buffer, bucket_size, effective_actual_len, unpad_outputs);
  auto t_detach_end = std::chrono::high_resolution_clock::now();

  if (!next_inputs.empty()) {
    int64_t next_actual = DetectActualLen(next_inputs, bundle.dynamic_specs_);
    int64_t next_bucket = bundle.SelectBucket(next_actual);
    st.pending_h2d = StartAsyncH2D(next_inputs, &ub.inputs[st.staging_buffer],
                                  ub.input_infos, pad_on_device, pool, dev,
                                  bundle.dynamic_specs_, ub.max_bucket_size,
                                  next_bucket, next_actual);
    st.prefetched_actual_len = next_actual;
    st.prefetched_bucket_size = next_bucket;
    std::swap(st.active_buffer, st.staging_buffer);
  } else {
    st.prefetched_actual_len = 0;
    st.prefetched_bucket_size = 0;
  }

  {
    double h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();
    double exec_ms = std::chrono::duration<double, std::milli>(t_exec_end - t_exec_start).count();
    double d2d_ms = std::chrono::duration<double, std::milli>(t_detach_end - t_detach_start).count();
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.total_h2d_time_ms += h2d_ms;
    stats_.total_nrt_exec_time_ms += exec_ms;
    stats_.total_d2h_time_ms += d2d_ms;
    stats_.total_executions++;
    stats_.bucket_hits[bucket_size]++;
    stats_.total_execution_time_ms += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();
  }

  return outputs;
}

void Engine::FlushPipeline(int64_t bundle_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = unified_pipeline_states_.find(bundle_id);
  if (it == unified_pipeline_states_.end()) return;
  auto& st = it->second;
  if (st.pending_h2d.valid()) st.pending_h2d.wait();
  st = UnifiedPipelineState{};
}

RuntimeStats Engine::GetStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void Engine::ResetStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_ = RuntimeStats{};
}

}  // namespace spiky

