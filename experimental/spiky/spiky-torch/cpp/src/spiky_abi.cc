#include "spiky_torch/abi.h"

#include <pybind11/pybind11.h>

#include <mutex>
#include <stdexcept>

namespace py = pybind11;

namespace spiky_torch {
namespace {

std::once_flag g_once;
const SpikyTorchABI* g_abi = nullptr;

}  // namespace

const SpikyTorchABI* GetABI() {
  if (!g_abi) {
    throw std::runtime_error("spiky_torch: ABI not loaded (import spiky.torch)");
  }
  return g_abi;
}

void LoadABIFromPython() {
  std::call_once(g_once, []() {
    py::gil_scoped_acquire gil;
    py::module_ spiky = py::module_::import("spiky");
    py::object cap = spiky.attr("_spiky").attr("_get_torch_abi")();
    if (!PyCapsule_CheckExact(cap.ptr())) {
      throw std::runtime_error("spiky_torch: spiky._get_torch_abi did not return a capsule");
    }
    g_abi = reinterpret_cast<const SpikyTorchABI*>(
        PyCapsule_GetPointer(cap.ptr(), kSpikyTorchCapsuleName));
    if (!g_abi || g_abi->abi_version != 1) {
      throw std::runtime_error("spiky_torch: ABI version mismatch");
    }
  });
}

}  // namespace spiky_torch

