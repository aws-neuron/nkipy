// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spiky/runtime.h"

#include "spiky/errors.h"

#include <atomic>
#include <mutex>

extern "C" {
#include <nrt/nrt.h>
#include <nrt/nrt_status.h>
}

namespace spiky {
namespace {

std::mutex g_mu;
std::atomic<bool> g_shutting_down{false};

static const char* NrtStatusStr(NRT_STATUS status) {
  switch (status) {
    case NRT_SUCCESS: return "NRT_SUCCESS";
    case NRT_FAILURE: return "NRT_FAILURE";
    case NRT_INVALID: return "NRT_INVALID";
    case NRT_INVALID_HANDLE: return "NRT_INVALID_HANDLE";
    case NRT_RESOURCE: return "NRT_RESOURCE";
    case NRT_TIMEOUT: return "NRT_TIMEOUT";
    case NRT_HW_ERROR: return "NRT_HW_ERROR";
    case NRT_QUEUE_FULL: return "NRT_QUEUE_FULL";
    case NRT_QUEUE_EMPTY: return "NRT_QUEUE_EMPTY";
    case NRT_LOAD_NOT_ENOUGH_NC: return "NRT_LOAD_NOT_ENOUGH_NC";
    case NRT_UNSUPPORTED_NEFF_VERSION: return "NRT_UNSUPPORTED_NEFF_VERSION";
    case NRT_FAIL_HOST_MEM_ALLOC: return "NRT_FAIL_HOST_MEM_ALLOC";
    case NRT_UNINITIALIZED: return "NRT_UNINITIALIZED";
    case NRT_CLOSED: return "NRT_CLOSED";
    case NRT_EXEC_BAD_INPUT: return "NRT_EXEC_BAD_INPUT";
    case NRT_EXEC_COMPLETED_WITH_NUM_ERR: return "NRT_EXEC_COMPLETED_WITH_NUM_ERR";
    case NRT_EXEC_COMPLETED_WITH_ERR: return "NRT_EXEC_COMPLETED_WITH_ERR";
    case NRT_EXEC_NC_BUSY: return "NRT_EXEC_NC_BUSY";
    case NRT_EXEC_OOB: return "NRT_EXEC_OOB";
    case NRT_COLL_PENDING: return "NRT_COLL_PENDING";
    case NRT_EXEC_HW_ERR_COLLECTIVES: return "NRT_EXEC_HW_ERR_COLLECTIVES";
    case NRT_EXEC_HW_ERR_HBM_UE: return "NRT_EXEC_HW_ERR_HBM_UE";
    case NRT_EXEC_HW_ERR_NC_UE: return "NRT_EXEC_HW_ERR_NC_UE";
    case NRT_EXEC_HW_ERR_DMA_ABORT: return "NRT_EXEC_HW_ERR_DMA_ABORT";
    case NRT_EXEC_SW_NQ_OVERFLOW: return "NRT_EXEC_SW_NQ_OVERFLOW";
    case NRT_EXEC_HW_ERR_REPAIRABLE_HBM_UE: return "NRT_EXEC_HW_ERR_REPAIRABLE_HBM_UE";
    default: return "UNKNOWN_STATUS";
  }
}

}  // namespace

Runtime& Runtime::Global() {
  static Runtime rt;
  return rt;
}

void Runtime::Init(int64_t device_id) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (refcount_ == 0) {
    InitImpl(device_id);
  }
  refcount_++;
}

void Runtime::Close() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (refcount_ <= 0) return;
  refcount_--;
  if (refcount_ == 0) {
    CloseImpl();
  }
}

void Runtime::InitImpl(int64_t device_id) {
  device_id_ = device_id;
  NRT_STATUS status = nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "spiky", "0.1.0");
  if (status != NRT_SUCCESS) {
    throw SpikyError(std::string("Failed to initialize NRT: ") + NrtStatusStr(status));
  }
  initialized_ = true;
  MarkReady();
}

void Runtime::CloseImpl() {
  MarkShuttingDown();
  // Best-effort; pool cleanup is explicit via empty_cache().
  nrt_close();
  initialized_ = false;
}

bool Runtime::IsInitialized() const { return initialized_; }
int64_t Runtime::DeviceId() const { return device_id_; }

int64_t Runtime::DeviceCount() const {
  uint32_t count = 0;
  NRT_STATUS status = nrt_get_visible_nc_count(&count);
  if (status != NRT_SUCCESS) return 0;
  return static_cast<int64_t>(count);
}

void Runtime::MarkShuttingDown() { g_shutting_down.store(true, std::memory_order_release); }
void Runtime::MarkReady() { g_shutting_down.store(false, std::memory_order_release); }
bool Runtime::IsShuttingDown() const { return g_shutting_down.load(std::memory_order_acquire); }

}  // namespace spiky

