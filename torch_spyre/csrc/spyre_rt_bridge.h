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
// The Rust bridge includes this header to see the deeptools types it passes
// through untouched, the csrc-owned HostKeepAlive, and the memory-pressure
// callback. Nothing in here may depend on flex.
//
// The bridge API itself (every type and call below the Runtime) comes from
// the cxx-GENERATED header of the spyre-torch crate — the same header the
// link probe linked (/work/phase0/linkprobe) — resolved from the crate's
// build output via setup.py's include dirs. It does not compile until
// spyre-torch is built; that is the contract, not an accident.

#pragma once

#include "spyre-torch/src/bridge.rs.h"

#include <spyrecode-host-functions/sendataconvert/sen_host_ops.h>
#include <spyrecode-host-functions/spyrecode.h>

#include <c10/core/Storage.h>

#include <memory>

namespace spyre {

// Holds a c10::Storage alive across an asynchronous copy whose host pages
// the Rust runtime touches after the call returns (a DCI-less H2D source, or
// a D2H destination). Defined here, in the csrc, because c10::Storage is a
// torch type the Rust side must not own. c10::Storage is itself an
// intrusive-ptr handle, so holding it by value is one refcount.
class HostKeepAlive {
 public:
  explicit HostKeepAlive(c10::Storage storage)
      : storage_(std::move(storage)) {}

 private:
  c10::Storage storage_;
};

// The vendor's FlexAllocator memory-pressure behaviour (flex_allocator.cpp
// allocateInDomain): release, run Python GC, retry. Called by the Rust
// allocator WITHOUT its lock held.
void on_memory_pressure();

}  // namespace spyre
