// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "nrt_wrapper.h"
#include "tensor_set.h"
#include <cstring>
#include <nrt/nrt_status.h>

extern "C" {

#define case_entry(x)                                                          \
  case x:                                                                      \
    return #x

const char *nrt_get_status_as_str(NRT_STATUS status) {
  switch (status) {
    case_entry(NRT_SUCCESS);
    case_entry(NRT_FAILURE);
    case_entry(NRT_INVALID);
    case_entry(NRT_INVALID_HANDLE);
    case_entry(NRT_RESOURCE);
    case_entry(NRT_TIMEOUT);
    case_entry(NRT_HW_ERROR);
    case_entry(NRT_QUEUE_FULL);
    case_entry(NRT_QUEUE_EMPTY);
    case_entry(NRT_LOAD_NOT_ENOUGH_NC);
    case_entry(NRT_UNSUPPORTED_NEFF_VERSION);
    case_entry(NRT_FAIL_HOST_MEM_ALLOC);
    case_entry(NRT_UNINITIALIZED);
    case_entry(NRT_CLOSED);
    case_entry(NRT_EXEC_BAD_INPUT);
    case_entry(NRT_EXEC_COMPLETED_WITH_NUM_ERR);
    case_entry(NRT_EXEC_COMPLETED_WITH_ERR);
    case_entry(NRT_EXEC_NC_BUSY);
    case_entry(NRT_EXEC_OOB);
    case_entry(NRT_COLL_PENDING);
    case_entry(NRT_EXEC_HW_ERR_COLLECTIVES);
    case_entry(NRT_EXEC_HW_ERR_HBM_UE);
    case_entry(NRT_EXEC_HW_ERR_NC_UE);
    case_entry(NRT_EXEC_HW_ERR_DMA_ABORT);
    case_entry(NRT_EXEC_SW_NQ_OVERFLOW);
    case_entry(NRT_EXEC_HW_ERR_REPAIRABLE_HBM_UE);
  default:
    return "UNKNOWN STATUS";
  }
}
}

namespace spike {

// NrtRuntime implementation
NrtRuntime::NrtRuntime(const std::string &fw_version,
                       const std::string &fal_version, bool owned)
    : initialized_(false), owned_(owned) {
  if (owned_) {
    NRT_STATUS status = nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, fw_version.c_str(),
                                 fal_version.c_str());
    if (status != 0) {
      throw NrtError(status, "Failed to initialize NRT runtime");
    }
  }
  initialized_ = true;
}

NrtRuntime::~NrtRuntime() {
  if (owned_ && initialized_) {
    nrt_close();
  }
}

uint32_t NrtRuntime::get_visible_nc_count() {
  uint32_t count = 0;
  NRT_STATUS status = nrt_get_visible_nc_count(&count);
  if (status != 0) {
    throw NrtError(status, "Failed to get visible NC count");
  }
  return count;
}

} // namespace spike
