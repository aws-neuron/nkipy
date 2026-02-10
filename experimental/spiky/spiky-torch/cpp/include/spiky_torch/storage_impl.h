#pragma once

#include <c10/core/StorageImpl.h>
#include <nrt/nrt.h>

namespace spiky_torch {

class SpikyStorageImpl : public c10::StorageImpl {
 public:
  using c10::StorageImpl::StorageImpl;

  nrt_tensor_t* nrt_tensor() const {
    return static_cast<nrt_tensor_t*>(data_ptr().get_context());
  }

  void release_resources() override;
};

}  // namespace spiky_torch

