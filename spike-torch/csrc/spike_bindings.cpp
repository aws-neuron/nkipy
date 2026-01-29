// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <c10/util/Logging.h>

#include "include/spike_guard_impl.h"
#include "include/spike_tensor_impl.h"
#include "include/spike_torch.h"

namespace spike_torch {

// Custom CPU fallback with warning system
void spike_cpu_fallback(const c10::OperatorHandle &op,
                        torch::jit::Stack *stack) {
  auto op_name = c10::toString(op.schema().operator_name());
  TORCH_WARN("CPU fallback operator: ", op_name);
  at::native::cpu_fallback(op, stack);
}

// Autograd fallback that redispatches to the actual implementation
void autograd_fallback(const c10::OperatorHandle &op,
                       c10::DispatchKeySet dispatch_keys,
                       torch::jit::Stack *stack) {
  auto new_keys = dispatch_keys & c10::after_autograd_keyset;
  op.redispatchBoxed(new_keys, stack);
}

// Register 'nkipy' as the name for PrivateUse1 device type
void register_spike_device() { c10::register_privateuse1_backend("nkipy"); }

} // namespace spike_torch

// Register the DeviceGuard implementation
C10_REGISTER_GUARD_IMPL(PrivateUse1, spike_torch::SpikeGuardImpl);

// Register CPU fallback for unimplemented operations
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &spike_torch::spike_cpu_fallback>());
}

// Register autograd fallback that just redispatches
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &spike_torch::autograd_fallback>());
}

// Register native functions with PyTorch dispatch
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // Tensor creation
  m.impl("empty.memory_format", &spike_torch::ops::empty_memory_format);
  m.impl("empty_strided", &spike_torch::ops::empty_strided);

  // Resize operations
  m.impl("resize_", &spike_torch::ops::resize_);

  // Copy operations
  m.impl("copy_", &spike_torch::ops::copy_);
  m.impl("_to_copy", &spike_torch::ops::_to_copy);
  m.impl("_copy_from_and_resize", &spike_torch::ops::_copy_from_and_resize);

  // View operations
  m.impl("view", &spike_torch::ops::view);
  m.impl("as_strided", &spike_torch::ops::as_strided);
  m.impl("_reshape_alias", &spike_torch::ops::_reshape_alias);

  // Fill operations
  m.impl("fill_.Scalar", &spike_torch::ops::fill_scalar);
}
