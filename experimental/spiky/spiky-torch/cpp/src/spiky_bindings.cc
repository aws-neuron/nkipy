// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <c10/util/Logging.h>

#include "spiky_torch/guard_impl.h"
#include "spiky_torch/tensor_impl.h"

namespace spiky_torch {

// CPU fallback with warning (data-movement only device).
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto op_name = c10::toString(op.schema().operator_name());
  TORCH_WARN("CPU fallback operator: ", op_name);
  at::native::cpu_fallback(op, stack);
}

void autograd_fallback(const c10::OperatorHandle& op,
                       c10::DispatchKeySet dispatch_keys,
                       torch::jit::Stack* stack) {
  auto new_keys = dispatch_keys & c10::after_autograd_keyset;
  op.redispatchBoxed(new_keys, stack);
}

void register_backend_name() { c10::register_privateuse1_backend("nkipy"); }

}  // namespace spiky_torch

// Register the DeviceGuard implementation
C10_REGISTER_GUARD_IMPL(PrivateUse1, spiky_torch::SpikyGuardImpl);

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&spiky_torch::cpu_fallback>());
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&spiky_torch::autograd_fallback>());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &spiky_torch::ops::empty_memory_format);
  m.impl("empty_strided", &spiky_torch::ops::empty_strided);
  m.impl("resize_", &spiky_torch::ops::resize_);
  m.impl("copy_", &spiky_torch::ops::copy_);
  m.impl("_to_copy", &spiky_torch::ops::_to_copy);
  m.impl("_copy_from_and_resize", &spiky_torch::ops::_copy_from_and_resize);
  m.impl("view", &spiky_torch::ops::view);
  m.impl("as_strided", &spiky_torch::ops::as_strided);
  m.impl("_reshape_alias", &spiky_torch::ops::_reshape_alias);
  m.impl("fill_.Scalar", &spiky_torch::ops::fill_scalar);
}

