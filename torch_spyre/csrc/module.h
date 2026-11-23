/*
 * Copyright 2025 The Torch-Spyre Authors.
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

#pragma once

#include <pybind11/pybind11.h>
#include <spyrecode-host-functions/sendataconvert/sen_host_ops.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>
#include <optional>

#include "spyre_rt_bridge.h"

using DataConversionStrideInfo = data_conversion_stride_info;
using DataConversionInfo = data_conversion_info;

namespace spyre {

// The process-wide spyre_rt runtime (spyre-torch crate). Created once by
// startRuntime(); the Box owns the Rust-side Runtime and every stream lives
// inside it. get() returns a borrowed pointer valid until freeRuntime().
class GlobalRuntime {
 public:
  static void set(rust::Box<spyre_rt::Runtime> runtime) {
    instance() = std::move(runtime);
  }

  static void reset() {
    instance().reset();
  }

  static spyre_rt::Runtime* get() {
    return instance() ? instance()->operator->() : nullptr;
  }

 private:
  GlobalRuntime() = delete;
  ~GlobalRuntime() = delete;

  static std::optional<rust::Box<spyre_rt::Runtime>>& instance() {
    static std::optional<rust::Box<spyre_rt::Runtime>> s;
    return s;
  }
};
bool get_downcast_warn_enabled();
bool get_hazard_tracker_enabled();
bool is_supported_dtype(c10::ScalarType dtype);
DataFormats get_device_dtype(c10::ScalarType torch_dtype);

int device_count();
void startRuntime();
}  // namespace spyre
