/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spyre_device_enum.h"

#include "spyre_rt.h"

namespace spyre {

int getVisibleDeviceCount() {
  // Get the number of available Spyre devices from the spyre_rt runtime.
  // Behaviour is the vendor flex::getNumDevices() (PF/VF: total devices in the
  // system, adjusted by AIU_WORLD_SIZE and SPYRE_DEVICES; MOCK: AIU_WORLD_SIZE,
  // error when < 1) — now answered by the spyre-torch crate.
  return static_cast<int>(spyre_rt::num_devices());
}

}  // namespace spyre
