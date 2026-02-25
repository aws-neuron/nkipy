#include "sys_trace.h"

namespace spike {

SysTraceGuard::SysTraceGuard(std::optional<uint32_t> core_id)
    : core_id_(core_id), started_(false) {
  nrt_sys_trace_config_t *config = nullptr;
  NRT_STATUS status = nrt_sys_trace_config_allocate(&config);
  if (status != NRT_SUCCESS) {
    throw NrtError(status, "Failed to allocate sys trace config");
  }

  if (core_id_.has_value()) {
    nrt_sys_trace_config_set_capture_enabled_for_nc(config, core_id_.value(),
                                                    true);
  }
  // When core_id is not set the default config captures all cores.

  status = nrt_sys_trace_start(config);
  nrt_sys_trace_config_free(config);

  if (status != NRT_SUCCESS) {
    throw NrtError(status, "Failed to start sys trace");
  }

  started_ = true;
}

SysTraceGuard::~SysTraceGuard() { stop(); }

void SysTraceGuard::stop() {
  if (started_) {
    nrt_sys_trace_stop();
    started_ = false;
  }
}

std::string SysTraceGuard::fetch_events_json() {
  if (!started_) {
    throw SpikeError("Cannot fetch events: sys trace is not started");
  }

  nrt_sys_trace_fetch_options_t *options = nullptr;
  NRT_STATUS status = nrt_sys_trace_fetch_options_allocate(&options);
  if (status != NRT_SUCCESS) {
    throw NrtError(status, "Failed to allocate fetch options");
  }

  if (core_id_.has_value()) {
    nrt_sys_trace_fetch_options_set_nc_idx(options, core_id_.value());
  }
  // When core_id is not set we skip set_nc_idx so NRT returns all cores.

  char *buffer = nullptr;
  size_t written_size = 0;
  status = nrt_sys_trace_fetch_events(&buffer, &written_size, options);
  nrt_sys_trace_fetch_options_free(options);

  if (status != NRT_SUCCESS) {
    if (buffer) {
      nrt_sys_trace_buffer_free(buffer);
    }
    throw NrtError(status, "Failed to fetch trace events");
  }

  std::string result;
  if (buffer && written_size > 0) {
    result.assign(buffer, written_size);
    nrt_sys_trace_buffer_free(buffer);
  }

  return result;
}

void SysTraceGuard::drain_events() {
  if (!started_) {
    throw SpikeError("Cannot drain events: sys trace is not started");
  }

  nrt_sys_trace_fetch_options_t *options = nullptr;
  NRT_STATUS status = nrt_sys_trace_fetch_options_allocate(&options);
  if (status != NRT_SUCCESS) {
    throw NrtError(status, "Failed to allocate fetch options");
  }

  if (core_id_.has_value()) {
    nrt_sys_trace_fetch_options_set_nc_idx(options, core_id_.value());
  }

  char *buffer = nullptr;
  size_t written_size = 0;
  status = nrt_sys_trace_fetch_events(&buffer, &written_size, options);
  nrt_sys_trace_fetch_options_free(options);

  if (buffer) {
    nrt_sys_trace_buffer_free(buffer);
  }

  if (status != NRT_SUCCESS) {
    throw NrtError(status, "Failed to drain trace events");
  }
}

} // namespace spike
