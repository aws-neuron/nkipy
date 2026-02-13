// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "spike.h"
#include <iostream>

namespace spike {

// Static guard to prevent multiple alive Spike instances
static bool g_alive_spike_instance_exists = false;

Spike::Spike(int verbose_level)
    : verbose_level_(verbose_level), runtime_(nullptr) {
  if (g_alive_spike_instance_exists) {
    throw SpikeError(
        "Cannot create Spike instance: a previous instance still exists. "
        "Call close() on the existing instance or delete spike with all "
        "tensors/models first.");
  }

  // RAII: Initialize NRT in constructor
  if (verbose_level_ > 0) {
    std::cout << "Initializing SPIKE Runtime" << std::endl;
  }

  runtime_ = std::make_unique<NrtRuntime>("spike", "1.0", true);
  g_alive_spike_instance_exists = true;
}

Spike::~Spike() {
  if (runtime_) {
    close();
  }
}

uint32_t Spike::get_visible_neuron_core_count() {
  return NrtRuntime::get_visible_nc_count();
}

int Spike::close() {
  runtime_.reset();
  g_alive_spike_instance_exists = false;
  return 0;
}

NrtModel Spike::load_model(const std::string &neff_file, uint32_t core_id,
                           bool cc_enabled, uint32_t rank_id,
                           uint32_t world_size) {
  return NrtModel(neff_file, core_id, cc_enabled, rank_id, world_size, this);
}

void Spike::unload_model(NrtModel &model) { model.unload(); }

NrtTensor Spike::allocate_tensor(size_t size, uint32_t core_id,
                                 std::optional<std::string> name) {
  std::string actual_name = name.value_or("unnamed_tensor");
  return NrtTensor(NRT_TENSOR_PLACEMENT_DEVICE, core_id, size, actual_name,
                   this);
}

NrtTensor Spike::slice_from_tensor(const NrtTensor &source, size_t offset,
                                   size_t size,
                                   std::optional<std::string> name) {
  size_t actual_size = (size == 0) ? (source.get_size() - offset) : size;
  std::string actual_name = name.value_or("unnamed_tensor");

  return NrtTensor(source, offset, actual_size, actual_name);
}

void Spike::free_tensor(NrtTensor &tensor) { tensor.free(); }

void Spike::tensor_write(NrtTensor &tensor, const void *data, size_t data_size,
                         size_t offset) {
  tensor.write(data, data_size, offset);
}

std::vector<uint8_t> Spike::tensor_read(const NrtTensor &tensor, size_t offset,
                                        size_t size) {
  size_t actual_size = (size == 0) ? (tensor.get_size() - offset) : size;

  std::vector<uint8_t> result(actual_size);
  tensor.read(result.data(), actual_size, offset);
  return result;
}

void Spike::tensor_write_from_pybuffer(NrtTensor &tensor, const void *data,
                                       size_t data_size, size_t offset) {
  tensor_write(tensor, data, data_size, offset);
}

NrtTensorSet Spike::create_tensor_sets(
    const std::unordered_map<std::string, NrtTensor &> &tensor_map) {
  NrtTensorSet tensor_set;

  for (const auto &[name, tensor] : tensor_map) {
    // FIXME: consider time-of-check-time-of-use (TOCTOU) race condition
    if (tensor.is_freed()) {
      throw SpikeError("Tensor '" + name +
                       "' is freed. Unable to add it into the tensor set. "
                       "Please check the lifetime of the tensor.");
    }
    tensor_set.add_tensor(name, tensor);
  }

  return tensor_set;
}

void Spike::execute(NrtModel &model,
                    const std::unordered_map<std::string, NrtTensor &> &inputs,
                    const std::unordered_map<std::string, NrtTensor &> &outputs,
                    std::optional<std::string> ntff_name, bool save_trace) {
  // Create tensor sets
  NrtTensorSet input_set = create_tensor_sets(inputs);
  NrtTensorSet output_set = create_tensor_sets(outputs);

  // Execute
  model.execute(input_set, output_set, ntff_name, save_trace);
}

std::string Spike::dtype_to_string(nrt_dtype_t dtype) {
  switch (dtype) {
  case NRT_DTYPE_FLOAT32:
    return "float32";
  case NRT_DTYPE_FLOAT16:
    return "float16";
  case NRT_DTYPE_BFLOAT16:
    return "bfloat16";
  case NRT_DTYPE_INT8:
    return "int8";
  case NRT_DTYPE_UINT8:
    return "uint8";
  case NRT_DTYPE_INT16:
    return "int16";
  case NRT_DTYPE_UINT16:
    return "uint16";
  case NRT_DTYPE_INT32:
    return "int32";
  case NRT_DTYPE_UINT32:
    return "uint32";
  case NRT_DTYPE_INT64:
    return "int64";
  case NRT_DTYPE_UINT64:
    return "uint64";
  default:
    return "unknown";
  }
}

ModelTensorInfo Spike::get_tensor_info(NrtModel &model) {
  // Get tensor info from NRT
  nrt_tensor_info_array_t *tensor_info = model.get_tensor_info();

  ModelTensorInfo result;

  // Process each tensor
  for (uint64_t i = 0; i < tensor_info->tensor_count; ++i) {
    const nrt_tensor_info_t &info = tensor_info->tensor_array[i];

    TensorMetadata metadata;
    metadata.size = info.size;
    metadata.dtype = dtype_to_string(info.dtype);

    if (info.shape && info.ndim > 0) {
      const uint32_t max_reasonable_ndim = 64;
      if (info.ndim <= max_reasonable_ndim) {
        metadata.shape =
            std::vector<int64_t>(info.shape, info.shape + info.ndim);
      } else {
        throw SpikeError("Tensor has unreasonably large ndim: " +
                         std::to_string(info.ndim));
      }
    }

    std::string tensor_name(info.name);
    if (info.usage == NRT_TENSOR_USAGE_INPUT) {
      result.inputs[tensor_name] = std::move(metadata);
    } else if (info.usage == NRT_TENSOR_USAGE_OUTPUT) {
      result.outputs[tensor_name] = std::move(metadata);
    }
  }

  // Free tensor info
  NrtModel::free_tensor_info(tensor_info);

  return result;
}

} // namespace spike
