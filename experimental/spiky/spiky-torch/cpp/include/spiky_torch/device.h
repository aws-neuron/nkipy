#pragma once

namespace spiky_torch {
namespace device {

int current_device();
void set_device(int device);
int device_count();
bool is_available();

}  // namespace device
}  // namespace spiky_torch

