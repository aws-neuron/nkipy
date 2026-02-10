#include "spiky_torch/hooks.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>

#include "spiky_torch/abi.h"
#include "spiky_torch/allocator.h"
#include "spiky_torch/device.h"

namespace spiky_torch {

class SpikyHooksInterface final : public at::PrivateUse1HooksInterface {
 public:
  void init() const override {}

  bool isPinnedPtr(const void* data) const override {
    if (!data) return false;
    const SpikyTorchABI* abi = GetABI();
    return abi->find_tensor(const_cast<void*>(data)) != nullptr;
  }

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    return device_index >= 0 &&
           device_index < static_cast<c10::DeviceIndex>(device::device_count());
  }

  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t new_bytes) const override {
    auto* alloc = storage.allocator();
    TORCH_CHECK(alloc != nullptr, "Storage must have an allocator");

    auto old_bytes = storage.nbytes();
    if (new_bytes == old_bytes) return;

    at::DataPtr new_data;
    if (new_bytes > 0) new_data = alloc->allocate(new_bytes);

    if (old_bytes > 0 && new_bytes > 0) {
      size_t copy_bytes = std::min(old_bytes, new_bytes);
      if (copy_bytes > 0 && new_data.get() && storage.data()) {
        const SpikyTorchABI* abi = GetABI();
        bool ok = abi->copy_by_ptr(new_data.get(), storage.data(), copy_bytes);
        TORCH_CHECK(ok, "resizePrivateUse1Bytes: failed to copy spiky storage");
      }
    }

    storage.set_data_ptr_noswap(std::move(new_data));
    storage.set_nbytes(new_bytes);
  }

  bool isAvailable() const override { return device::device_count() > 0; }
};

static SpikyHooksInterface g_hooks;
static bool g_registered = false;

void register_hooks() {
  if (!g_registered) {
    at::RegisterPrivateUse1HooksInterface(&g_hooks);
    g_registered = true;
  }
}

}  // namespace spiky_torch

