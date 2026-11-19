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

// The C++ side of the spyre_rt cxx bridge (docs/torch-spyre/API.md).
// The bridge API itself (every type and call below the Runtime) comes from
// the cxx-GENERATED header of the spyre-torch crate — the same header the
// link probe linked (/work/phase0/linkprobe) — resolved from the crate's
// build output via setup.py's include dirs. The deeptools includes below are
// the types the bridge passes through untouched (data_conversion_info, Hcm);
// the keep-alive BASE class and the memory-pressure callback are declared by
// the crate's own include/spyre_torch.h, which the generated header pulls
// in. Nothing in here may depend on flex.

#pragma once

#include "spyre-torch/src/bridge.rs.h"

#include <spyrecode-host-functions/sendataconvert/sen_host_ops.h>
#include <spyrecode-host-functions/spyrecode.h>

#include <c10/core/Storage.h>

#include <memory>

namespace spyre {

// The csrc's keep-alive for a copy whose host pages the Rust runtime
// touches after the call returns (a raw H2D source, or a D2H destination):
// spyre_rt::HostKeepAlive (the crate's base class) holding a c10::Storage.
// Defined here, in the csrc, because c10::Storage is a torch type the Rust
// side must not own; it is an intrusive-ptr handle, so holding it by value
// is one refcount. The runtime destroys it when the transfer completes or
// fails — possibly on the runtime's completion thread — so nothing here may
// touch the GIL or torch's allocator lock.
class StorageKeepAlive : public spyre_rt::HostKeepAlive {
 public:
  explicit StorageKeepAlive(c10::Storage storage)
      : storage_(std::move(storage)) {}

 private:
  c10::Storage storage_;
};

}  // namespace spyre
