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

// ============================================================================
// STUB — throw-only implementations of the spyre_rt bridge
// (spyre_rt.h). Exists so the patched csrc compiles and links before the
// spyre-torch crate lands. EVERY entry point throws; nothing here touches a
// device. Deleted together with spyre_rt.h when the real cxx bridge arrives.
// ============================================================================

#include "spyre_rt.h"

#include <stdexcept>

namespace spyre_rt {
namespace {
[[noreturn]] void stub(const char* fn) {
  throw std::runtime_error(std::string("spyre_rt stub: ") + fn +
                           " is not implemented (spyre-torch crate not linked)");
}
}  // namespace

uint32_t num_devices() { stub("num_devices"); }
rust::Box<Runtime> create_runtime(DeviceIndex) { stub("create_runtime"); }

const Stream& Runtime::default_stream() const { stub("default_stream"); }
const Stream& Runtime::create_stream(StreamPriority, StreamOrdering,
                                     bool) const {
  stub("create_stream");
}
bool Runtime::has_stream_error() const { stub("has_stream_error"); }
rust::Box<Allocation> Runtime::allocate(ByteSize, MemoryType,
                                        rust::Slice<const DomainId>) const {
  stub("allocate");
}
MemoryStats Runtime::memory_stats(MemoryType) const {
  stub("memory_stats");
}
uint32_t Runtime::num_domains() const { stub("num_domains"); }
uint64_t Runtime::device_address(DeviceView) const { stub("device_address"); }
rust::Box<Corrections> Runtime::host_compute(
    const HostComputeHandle&, ByteSize, const uint8_t*,
    rust::Slice<const HostComputeArg>) const {
  stub("host_compute");
}

void Stream::synchronize() const { stub("synchronize"); }
bool Stream::query() const { stub("query"); }
void Stream::copy_h2d(const uint8_t*, ByteSize, DeviceView,
                      std::unique_ptr<spyre::HostKeepAlive>,
                      PipelineBarrier) const {
  stub("copy_h2d");
}
void Stream::copy_h2d_converted(const uint8_t*, ByteSize, DeviceView,
                                const data_conversion_info&,
                                PipelineBarrier) const {
  stub("copy_h2d_converted");
}
void Stream::copy_d2h(DeviceView, uint8_t*, ByteSize,
                      std::unique_ptr<spyre::HostKeepAlive>,
                      PipelineBarrier) const {
  stub("copy_d2h");
}
void Stream::copy_d2h_converted(DeviceView, uint8_t*, ByteSize,
                                const data_conversion_info&,
                                std::unique_ptr<spyre::HostKeepAlive>,
                                PipelineBarrier) const {
  stub("copy_d2h_converted");
}
void Stream::upload_corrections(rust::Box<Corrections>, DeviceView,
                                PipelineBarrier) const {
  stub("upload_corrections");
}
void Stream::launch_compute(DeviceView, BootstrapOffset,
                            rust::Slice<const DeviceView>,
                            PipelineBarrier) const {
  stub("launch_compute");
}
size_t Stream::max_program_corrections() const { return 0; }
void Stream::fill(DeviceView, double, DataFormat) const { stub("fill"); }

DeviceView Allocation::view() const { stub("Allocation::view"); }

DecodedPointer decode_device_pointer(uint64_t) {
  stub("decode_device_pointer");
}
rust::Box<HostComputeHandle> create_host_compute_handle(std::unique_ptr<Hcm>) {
  stub("create_host_compute_handle");
}
ByteSize size(const Corrections&) { stub("size"); }
rust::String to_string(DeviceView) { stub("to_string"); }

}  // namespace spyre_rt
