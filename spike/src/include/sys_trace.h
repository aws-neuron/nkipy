#ifndef SPIKE_SRC_INCLUDE_SYS_TRACE_H
#define SPIKE_SRC_INCLUDE_SYS_TRACE_H

#include "error.h"
#include <nrt/nrt_sys_trace.h>

#include <cstdint>
#include <optional>
#include <string>

namespace spike {

// RAII wrapper for NRT system trace capture.
// If core_id is given, traces that single core.
// If omitted, traces all visible NeuronCores.
class SysTraceGuard {
public:
  explicit SysTraceGuard(std::optional<uint32_t> core_id = std::nullopt);
  ~SysTraceGuard();

  // Non-copyable, non-movable
  SysTraceGuard(const SysTraceGuard &) = delete;
  SysTraceGuard &operator=(const SysTraceGuard &) = delete;
  SysTraceGuard(SysTraceGuard &&) = delete;
  SysTraceGuard &operator=(SysTraceGuard &&) = delete;

  // Stop tracing (idempotent). Called automatically by destructor.
  void stop();

  // Fetch events as JSON string, consumes events from the ring buffer
  std::string fetch_events_json();

  // Fetch and discard events (clears the ring buffer)
  void drain_events();

private:
  std::optional<uint32_t> core_id_;
  bool started_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_SYS_TRACE_H
