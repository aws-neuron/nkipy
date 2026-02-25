#ifndef SPIKE_SRC_INCLUDE_ERROR_H
#define SPIKE_SRC_INCLUDE_ERROR_H

#include <nrt/nrt.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace spike {

class SpikeRuntimeError : public std::runtime_error {
public:
  SpikeRuntimeError(const std::string &message) : std::runtime_error(message) {}
};

class SpikeError : public SpikeRuntimeError {
public:
  SpikeError(std::string message)
      : SpikeRuntimeError(std::string("Spike Error: ") + message) {}
};

class NrtError : public SpikeRuntimeError {
public:
  NrtError(NRT_STATUS error_code, std::string message = "")
      : SpikeRuntimeError([error_code, &message]() {
          std::ostringstream oss;
          oss << "NRT Error " << nrt_get_status_as_str(error_code) << "("
              << error_code << ")";

          if (message != "") {
            oss << ": " << message;
          }

          return oss.str();
        }()),
        error_code_(error_code) {}

  NRT_STATUS error_code() const { return error_code_; }

private:
  NRT_STATUS error_code_;
};

} // namespace spike

#endif // SPIKE_SRC_INCLUDE_ERROR_H
