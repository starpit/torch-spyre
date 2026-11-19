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
// STUB — temporary C++ stand-in for the cxx-generated header of the
// `spyre-torch` crate (docs/torch-spyre/API.md, design v1).
//
// The signatures below mirror what cxx generates for the bridge: shared
// structs/enums by value, `rust::Box` for Rust opaques, `Result<T>` as
// "returns T, throws rust::Error", `&[T]` as `rust::Slice<const T>`,
// `UniquePtr<C++ type>` as `std::unique_ptr`. When the crate lands, this file
// is deleted and the generated header takes its place; csrc call sites do not
// change. Implementations live in spyre_rt_stub.cpp (also deleted then).
// ============================================================================

#pragma once

#include <rust/cxx.h>

#include <cstdint>
#include <memory>

#include "spyre_rt_bridge.h"

namespace spyre_rt {

// ---- shared value types: POD, Copy, the same layout on both sides ----
struct DeviceIndex {
  uint32_t raw;
};
struct RegionId {
  uint64_t raw;
};
struct DomainId {
  uint32_t raw;
};
struct ByteOffset {
  uint64_t raw;
};
struct ByteSize {
  uint64_t raw;
};
struct BootstrapOffset {
  uint64_t raw;
};

/// One contiguous device range. Replaces flex Chunk + LogicalAddress for every
/// use the csrc makes: offset views, D2H narrowing, program offsets.
struct DeviceView {
  RegionId region;
  ByteOffset offset;
  ByteSize size;
  DomainId domain;
};

enum class MemoryType : uint32_t { Tensor, Program };
enum class StreamPriority : uint32_t { Normal, High };
enum class StreamOrdering : uint32_t { Strict };
enum class PipelineBarrier : uint32_t { No, Yes };

/// The deeptools DataFormats values the csrc emits, same discriminants
/// (deeptools util/sendefs/sendefs.h enum class DataFormats).
enum class DataFormat : uint32_t {
  SEN169_FP16,
  IEEE_FP32,
  INVALID,
  SEN143_FP8,
  SEN152_FP8,
  SEN153_FP9,
  SENINT2,
  SENINT4,
  SENINT8,
  SENINT16,
  SENINT24,
  IEEE_INT64,
  IEEE_INT32,
  SENUINT32,
  SENUINT2,
  IEEE_FP16,
  BOOL,
  BFLOAT16,
  SEN18F_FP24,
  SEN080_FP8,
  SEN053_FP8,
  SEN121_FP4,
  SENUINT8,
};

struct MemoryStats {
  ByteSize free;
  ByteSize total;
};
struct DecodedPointer {
  uint32_t segment;
  ByteOffset offset;
  MemoryType memory_type;
};

/// Host-compute argument: an address, or a scalar. Tag plus payload.
enum class HostComputeArgKind : uint32_t { Address, Value };
struct HostComputeArg {
  HostComputeArgKind kind;
  DeviceView view;
  int64_t value;
};

// ---- Rust opaques ----
class Runtime;
class Stream;
class Allocation;         // owning; dropping it frees the device block
class HostComputeHandle;  // owns the deeptools Hcm
class Corrections;        // host-compute output bytes, pooled

// A. start
uint32_t num_devices();
rust::Box<Runtime> create_runtime(DeviceIndex device);

class Runtime {
 public:
  // B. streams (they live as long as the Runtime)
  const Stream& default_stream() const;
  const Stream& create_stream(StreamPriority p, StreamOrdering o,
                              bool track_hazards) const;
  // C. sync
  bool has_stream_error() const;
  // D. memory
  rust::Box<Allocation> allocate(ByteSize size, MemoryType ty,
                                 rust::Slice<const DomainId> domains) const;
  MemoryStats memory_stats(MemoryType ty) const;
  uint32_t num_domains() const;
  uint64_t device_address(DeviceView v) const;  // sharded path only
  // G. launch (host compute runs inline on the caller, like the vendor's)
  rust::Box<Corrections> host_compute(
      const HostComputeHandle& h, ByteSize correction_size,
      const uint8_t* input, rust::Slice<const HostComputeArg> args) const;
};

class Stream {
 public:
  // C. sync
  void synchronize() const;  // rethrows a deferred error once
  bool query() const;
  // E. copies. Two entry points per direction: the csrc decides identity vs
  //    converting (dci_is_identity_copy stays csrc-side). The dci is borrowed
  //    for the call only.
  void copy_h2d(const uint8_t* src, ByteSize src_capacity, DeviceView dst,
                std::unique_ptr<spyre::HostKeepAlive> keep,
                PipelineBarrier barrier) const;
  void copy_h2d_converted(const uint8_t* src, ByteSize src_capacity,
                          DeviceView dst, const data_conversion_info& dci,
                          PipelineBarrier barrier) const;
  void copy_d2h(DeviceView src, uint8_t* dst, ByteSize dst_capacity,
                std::unique_ptr<spyre::HostKeepAlive> keep,
                PipelineBarrier barrier) const;
  void copy_d2h_converted(DeviceView src, uint8_t* dst, ByteSize dst_capacity,
                          const data_conversion_info& dci,
                          std::unique_ptr<spyre::HostKeepAlive> keep,
                          PipelineBarrier barrier) const;
  // G. launch
  void upload_corrections(rust::Box<Corrections> c, DeviceView dst,
                          PipelineBarrier barrier) const;
  void launch_compute(DeviceView program, BootstrapOffset bootstrap,
                      rust::Slice<const DeviceView> tensors,
                      PipelineBarrier barrier) const;
  // PF returns 0, the vendor base default (runtime_scheduler.hpp:141);
  // batching is VF-only.
  size_t max_program_corrections() const;
  // H. fill
  void fill(DeviceView dst, double value, DataFormat format) const;
};

class Allocation {
 public:
  DeviceView view() const;
};

// F. program addresses
DecodedPointer decode_device_pointer(uint64_t raw);

// G. launch
rust::Box<HostComputeHandle> create_host_compute_handle(std::unique_ptr<Hcm> hcm);
ByteSize size(const Corrections& c);

// Debug-only Display for DeviceView (the vendor's operator<<).
rust::String to_string(DeviceView v);

}  // namespace spyre_rt
