#pragma once

#include <nrt/nrt.h>
#include <nrt/nrt_status.h>
#include <cstddef>

static bool g_nrt_initialized = false;

static inline int nrt_util_init() {
  if (g_nrt_initialized) {
    return 0;
  }
  NRT_STATUS status = nrt_init(NRT_FRAMEWORK_TYPE_PYTORCH, "2.0", "1.0");
  if (status != NRT_SUCCESS) {
    return status;
  }
  g_nrt_initialized = true;
  return 0;
}

static inline int nrt_util_allocate_device_tensor(int nc_idx, size_t size_bytes,
                                                  void** out_address,
                                                  nrt_tensor_t** out_tensor) {
  int init_status = nrt_util_init();
  if (init_status != 0) {
    return init_status;
  }
  NRT_STATUS status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, nc_idx,
                                          size_bytes, "nrt_tensor", out_tensor);
  if (status != NRT_SUCCESS) {
    return status;
  }
  *out_address = nrt_tensor_get_va(*out_tensor);
  return 0;
}

static inline void nrt_util_free_tensor(nrt_tensor_t** tensor) {
  if (tensor && *tensor) {
    nrt_tensor_free(tensor);
  }
}
