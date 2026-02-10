// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spiky_torch/storage_impl.h"

namespace spiky_torch {

void SpikyStorageImpl::release_resources() {
  StorageImpl::release_resources();
}

}  // namespace spiky_torch

