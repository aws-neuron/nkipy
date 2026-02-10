#pragma once

#include <stdexcept>
#include <string>

namespace spiky {

inline std::string NrtStatusToString(int status) {
  // Keep this header-only and minimal; full mapping lives in runtime.cc.
  return std::to_string(status);
}

class SpikyError : public std::runtime_error {
 public:
  explicit SpikyError(const std::string& msg) : std::runtime_error(msg) {}
};

}  // namespace spiky

