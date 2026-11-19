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

#include "spyre_stream.h"

#include <ATen/record_function.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "job_plan.h"
#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_composite_address.h"
#include "spyre_error.h"
#include "spyre_guard.h"
#include "spyre_mem.h"
#include "spyre_rt_bridge.h"
#include "spyre_tensor_impl.h"

namespace spyre {
namespace {

// Bound the device view to the whole sticks covering the copied region, then
// rebase the DCI onto that staging range. This is the vendor's
// narrowD2HRange, restated as DeviceView arithmetic: when the transfer spans
// this range from offset zero, the runtime can reuse the DCI instead of
// constructing another compact conversion. The runtime can still compact
// strided selections within the bounded range. For a final-token logits
// view, the range is one row even when the tensor owns a complete prefill
// allocation.
std::optional<spyre_rt::DeviceView> narrowD2HRange(
    DataConversionInfo& dci, const SpyreTensorLayout& layout,
    const spyre_rt::DeviceView& allocation) {
  if (layout.element_arrangement != ElementArrangement::STANDARD ||
      (layout.device_dtype != DataFormats::SEN169_FP16 &&
       layout.device_dtype != DataFormats::IEEE_FP32 &&
       layout.device_dtype != DataFormats::IEEE_INT32) ||
      dci.dcsi_.empty() || dci.input_shape_.empty()) {
    return std::nullopt;
  }

  const int64_t stick_elements = layout.elems_per_stick();
  const int64_t element_bytes = 128 / stick_elements;
  const int64_t allocation_elements =
      static_cast<int64_t>(allocation.size.raw) / element_bytes;
  int64_t first = allocation_elements;
  int64_t last = 0;
  for (const auto& stride : dci.dcsi_) {
    int64_t end = stride.offset_src_;
    if (end < 0 || end >= allocation_elements) {
      return std::nullopt;
    }
    for (size_t i = 0; i < stride.size_.size(); ++i) {
      const int64_t count = stride.size_[i];
      const int64_t step = stride.stride_src_[i];
      // Check against the allocation before multiplying to avoid overflow.
      if (count <= 0 || step < 0 ||
          (step > 0 && count - 1 > (allocation_elements - 1 - end) / step)) {
        return std::nullopt;
      }
      end += (count - 1) * step;
    }
    first = std::min(first, stride.offset_src_);
    last = std::max(last, end + 1);
  }
  first = first / stick_elements * stick_elements;
  last = (last + stick_elements - 1) / stick_elements * stick_elements;
  if (last > allocation_elements || last <= first ||
      (first == 0 && last == allocation_elements)) {
    return std::nullopt;
  }

  // Single-view STANDARD storage is a dense sequence of 128-byte sticks.
  // get_device_stride_infos builds contiguous physical device strides, so DCI
  // offsets already include layout padding; no extra inter-stick/tile gaps
  // need to be added when converting physical element offsets to byte
  // offsets.
  for (auto& stride : dci.dcsi_) {
    stride.offset_src_ -= first;
  }
  // Describe the flat staged range while preserving the original DCI rank.
  // Shapes are innermost-first: collapse the outer dimensions to 1 so the
  // product remains the staged element count. dcsi_ retains the view strides.
  auto& shape = dci.input_shape_;
  std::fill(shape.begin() + 1, shape.end(), 1);
  shape.front() = last - first;
  return spyre_rt::DeviceView{
      allocation.region,
      spyre_rt::ByteOffset{allocation.offset.raw +
                           static_cast<uint64_t>(first) * element_bytes},
      spyre_rt::ByteSize{static_cast<uint64_t>(last - first) * element_bytes},
      allocation.domain};
}

// TODO(tmhoangt): torch-spyre manages the pool and mapping; the runtime just
// creates/destroys individual streams when asked.

// Global stream pool (shared across all threads)
struct StreamPool {
  mutable std::shared_mutex mutex;

  // Per-device stream pools
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      low_priority_streams;
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      high_priority_streams;
  std::unordered_map<c10::DeviceIndex, std::vector<c10::StreamId>>
      host_compute_streams;

  // Round-robin indices
  std::unordered_map<c10::DeviceIndex, size_t> next_low_priority_idx;
  std::unordered_map<c10::DeviceIndex, size_t> next_high_priority_idx;
  std::unordered_map<c10::DeviceIndex, size_t> next_host_compute_idx;

  // Mapping from c10::StreamId to the runtime's Stream. The streams live as
  // long as the GlobalRuntime; the pool stores non-owning pointers.
  // NOTE: this assumes one Spyre device per process (PyTorch's model for
  // multi-device use is one process per device), so the map is keyed by stream
  // id only. Supporting multiple devices in a single process would require
  // keying by (c10::DeviceIndex, c10::StreamId) so each device has its own
  // handles.
  std::unordered_map<c10::StreamId, const spyre_rt::Stream*> stream_handle_map;

  // Per-device initialization flags
  std::unordered_map<c10::DeviceIndex, std::once_flag> device_init_flags;

  // Set to true once initializeStreamPoolImpl has been called.
  bool initialized = false;
  // Records the device_index passed to initializeStreamPoolImpl (valid only
  // when initialized == true), used in error messages.
  c10::DeviceIndex initialized_device_index = -1;
};

StreamPool& getStreamPool() {
  static StreamPool pool;
  return pool;
}

thread_local std::unordered_map<c10::DeviceIndex, c10::StreamId>
    current_streams;

}  // anonymous namespace

// Stream pool configuration
// Per device:
// - Stream 0: Default stream (always available, priority 0)
// - Streams 1-32: Low priority streams (priority 0)
// - Streams 33-64: High priority streams (priority -1)
// - Stream 65+: Host compute streams (priority -1)
constexpr int kStreamsPerDevice = 32;
constexpr int kHighPriorityStreamsPerDevice = 32;
constexpr int kHostComputeStreamStartPerDevice = 65;
constexpr int kDefaultHostComputeStreams = 4;
constexpr int kMaxHostComputeStreams = 8;

// Read from environment variable, defaulting to kDefaultHostComputeStreams when
// unset. An explicit value less than 1 (including malformed input, which parses
// to 0) is an error; values above kMaxHostComputeStreams are capped.
inline int getNumHostComputeStreams() {
  const char* env = std::getenv("TORCH_SPYRE_NUM_HOST_COMPUTE_STREAMS");
  int n = env ? std::atoi(env) : kDefaultHostComputeStreams;
  TORCH_CHECK(n >= 1,
              "TORCH_SPYRE_NUM_HOST_COMPUTE_STREAMS must be at least 1, got '",
              env, "'");
  if (n > kMaxHostComputeStreams) n = kMaxHostComputeStreams;
  return n;
}

// Constructor
SpyreStream::SpyreStream()
    : stream_(getCurrentStream(c10::Device(c10::DeviceType::PrivateUse1,
                                           SpyreGuardImpl::tls_idx))
                  .unwrap()) {}
SpyreStream::SpyreStream(c10::Stream stream) : stream_(stream) {
  TORCH_CHECK(stream_.device_type() == c10::DeviceType::PrivateUse1,
              "SpyreStream requires PrivateUse1 device type, got ",
              stream_.device_type());
}

c10::StreamId SpyreStream::id() const {
  return stream_.id();
}

c10::Device SpyreStream::device() const {
  return stream_.device();
}

int SpyreStream::priority() const {
  // Determine priority from stream ID
  if (id() <= kStreamsPerDevice) {
    return 0;  // Low priority: stream 0 (default) and streams 1-32
  } else {
    return -1;
  }
}

bool SpyreStream::query() const {
  c10::DeviceGuard guard(stream_.device());

  SPYRE_RUNTIME_DEBUG() << "stream " << id() << " on device "
                        << static_cast<int>(device().index());

  return resolveRuntimeHandle()->query();
}

void SpyreStream::synchronize() const {
  RECORD_FUNCTION("host::synchronize", {});
  c10::DeviceGuard device_guard(stream_.device());

  SPYRE_RUNTIME_DEBUG() << "stream " << id() << " on device "
                        << static_cast<int>(device().index());

  resolveRuntimeHandle()->synchronize();
}

c10::Stream SpyreStream::unwrap() const {
  return stream_;
}

void SpyreStream::copyProgramAsync(
    void* prog_cpu_ptr, const spyre_rt::DeviceView* device_address) const {
  // NOTE: the assumption is that the size of the program matches the size of
  // the device allocation, so the buffer provably covers the DMA: pass the
  // allocation size as the host capacity (the vendor's plain factory has no
  // capacity, and its validators check dma_size <= capacity only).
  copyAsyncImpl(prog_cpu_ptr, /*host_capacity=*/device_address->size.raw,
                device_address, nullptr,
                /*host2device=*/true);
}

void SpyreStream::copyAsync(const at::Tensor& src,
                            const at::Tensor& dst) const {
  SPYRE_RUNTIME_DEBUG() << "src (" << src.scalar_type()
                        << ") is on:" << src.device();
  SPYRE_RUNTIME_DEBUG() << "dst (" << dst.scalar_type()
                        << ") on:" << dst.device();

  // Determine copy direction
  bool host2device = src.is_cpu() && dst.is_privateuseone();
  bool device2host = src.is_privateuseone() && dst.is_cpu();

  const at::Tensor* dev_tensor = host2device ? &dst : &src;
  const at::Tensor* cpu_tensor = host2device ? &src : &dst;

  if (host2device || device2host) {
    // DCI offsets are relative to storage, so pass the storage base and its
    // complete accessible extent rather than the tensor's logical view.
    void* cpu_ptr = const_cast<void*>(cpu_tensor->storage().data());
    const size_t host_capacity = cpu_tensor->storage().nbytes();

    // Get SpyreTensorLayout using the public API
    SpyreTensorLayout stl = get_spyre_tensor_layout(*dev_tensor);

    DataConversionInfo dci =
        generate_dci(cpu_tensor, dev_tensor, stl, host2device);

    // A DCI that describes a dense same-format copy is a no-op conversion.
    // Dropping it skips the runtime's host-side staging (deeptools::ConvertData
    // runs a per-element strided loop even for identical formats) and, on H2D,
    // keeps a pinned source buffer attached -- the staged-source path discards
    // it when a conversion stages into a fresh pageable buffer.
    //
    // The skip is only safe when the copied region is the whole device
    // allocation: the raw DMA is sized from the allocation's size, so a dense
    // leading slice (e.g. d[:64].cpu()) would otherwise transfer past the end
    // of the smaller host buffer. Comparing the host storage size is not
    // sufficient either -- a same-sized host buffer backing a slice view
    // (y[:4096].copy_(d[:4096]) on an 8192-element y) would let the
    // whole-allocation DMA overwrite the rest of y. Require the copied region
    // extents themselves to match the allocation on both sides. Slices keep
    // their DCI and go through the conversion path, which carries the region
    // extents.
    //
    // The host-side and device-side extent checks are currently equivalent:
    // spyre_copy_ rejects mismatched region sizes up front
    // (TORCH_CHECK(cpu_sizes == dev_sizes)), and dci_is_identity_copy
    // requires matching data formats (hence equal element sizes), so the two
    // products are always equal here. Both halves are kept anyway as
    // defense-in-depth, in case the size precondition is ever relaxed or
    // copyAsync gains a caller that bypasses spyre_copy_'s check.
    //
    // On big-endian hosts the device is little-endian, and a raw DMA skips
    // the byte-order conversion that deeptools::ConvertData performs, so
    // multi-byte elements would arrive byte-swapped. Single-byte elements
    // have no byte order, so they may still take the raw-DMA fast path.
    // dci_is_identity_copy requires matching data formats, so the CPU and
    // device element sizes are equal here.
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    const bool byte_order_safe = cpu_tensor->element_size() == 1;
#else
    const bool byte_order_safe = true;
#endif
    const auto* allocation = get_composite_address(*dev_tensor);
    const size_t dev_total_size = static_cast<size_t>(allocation->size.raw);
    const DataConversionInfo* dci_arg =
        (byte_order_safe && dci_is_identity_copy(dci) &&
         static_cast<size_t>(cpu_tensor->numel() *
                             cpu_tensor->element_size()) == dev_total_size &&
         static_cast<size_t>(dev_tensor->numel() *
                             dev_tensor->element_size()) == dev_total_size)
            ? nullptr
            : &dci;

    // The narrowed-range path is a conversion-path optimization: it rebases
    // the DCI onto the sticks covering the view. It has nothing to offer a
    // stripped identity copy (the region is the whole allocation by then),
    // so only take it when the DCI is kept.
    if (device2host && dci_arg != nullptr) {
      if (auto range = narrowD2HRange(dci, stl, *allocation)) {
        // The runtime clones the descriptor fields it needs into the
        // transfer's typed descriptor when enqueuing; the tensor continues
        // to own the underlying allocation, just as in the full-storage
        // path.
        copyAsyncImpl(cpu_ptr, host_capacity, &*range, &dci, host2device,
                      std::make_unique<spyre::StorageKeepAlive>(
                          cpu_tensor->storage()));
        return;
      }
    }

    // Hold the host storage alive across the DMA when the runtime will touch
    // the user pages after the call returns: that is every D2H (the raw DMA
    // or the conversion's write-back runs on the scheduler thread) and any
    // DCI-less H2D (with the DCI stripped there is no caller-thread staging
    // copy, so the scheduler thread reads the source pages directly). Hold
    // c10::Storage rather than at::Tensor: a tensor copy shares the
    // TensorImpl, which resize_/set_ mutate in place by swapping in a new
    // storage -- the old pages would be freed under the in-flight DMA.
    std::unique_ptr<spyre_rt::HostKeepAlive> keep;
    if (device2host || dci_arg == nullptr) {
      keep = std::make_unique<spyre::StorageKeepAlive>(cpu_tensor->storage());
    }
    copyAsyncImpl(cpu_ptr, host_capacity, allocation, dci_arg, host2device,
                  std::move(keep));

  } else {
    TORCH_CHECK(false, "Unsupported copy types: src on ", src.device(),
                " dst on ", dst.device());
  }
}

const spyre_rt::Stream* SpyreStream::resolveRuntimeHandle() const {
  auto& pool = getStreamPool();
  std::shared_lock<std::shared_mutex> lock(pool.mutex);

  auto it = pool.stream_handle_map.find(id());
  TORCH_CHECK(it != pool.stream_handle_map.end(),
              "SpyreStream: no runtime handle for stream id ", id(),
              " — was the stream pool initialized for this device?");
  return it->second;
}

SpyreStreamError SpyreStream::getError() const {
  // The vendor rolls per-stream needsShutdown() up against the runtime's
  // aggregate hasStreamError(); the two agree while both reduce to
  // needsShutdown() (module.cpp note on SpyreGetDeviceState).
  auto* runtime = GlobalRuntime::get();
  if (!runtime) {
    return SpyreStreamError::Success;
  }
  return runtime->has_stream_error() ? SpyreStreamError::Shutdown
                                     : SpyreStreamError::Success;
}

void SpyreStream::copyAsyncImpl(
    void* cpu_ptr, size_t host_capacity,
    const spyre_rt::DeviceView* device_address, const DataConversionInfo* dci,
    bool host2device,
    std::unique_ptr<spyre_rt::HostKeepAlive> keep) const {
  // The vendor's pageable path: createDmaParamsWithHostCapacity semantics
  // (the caller states the accessible host extent; the runtime bounds the
  // DMA and, on D2H with a DCI, bounds the conversion output by it). The
  // pinned-buffer fast path is not on the chat path and is not ported.
  const spyre_rt::Stream* handle = resolveRuntimeHandle();
  const spyre_rt::ByteSize capacity{host_capacity};
  if (host2device) {
    if (dci != nullptr) {
      handle->copy_h2d_converted(static_cast<const uint8_t*>(cpu_ptr),
                                 capacity, *device_address, *dci,
                                 spyre_rt::PipelineBarrier::No);
    } else {
      handle->copy_h2d(static_cast<const uint8_t*>(cpu_ptr), capacity,
                       *device_address, std::move(keep),
                       spyre_rt::PipelineBarrier::No);
    }
  } else {
    if (dci != nullptr) {
      handle->copy_d2h_converted(*device_address, static_cast<uint8_t*>(cpu_ptr),
                                 capacity, *dci, std::move(keep),
                                 spyre_rt::PipelineBarrier::No);
    } else {
      handle->copy_d2h(*device_address, static_cast<uint8_t*>(cpu_ptr),
                       capacity, std::move(keep),
                       spyre_rt::PipelineBarrier::No);
    }
  }
}

void SpyreStream::copyH2D(const uint8_t* src, spyre_rt::ByteSize src_capacity,
                          spyre_rt::DeviceView dst,
                          spyre_rt::PipelineBarrier barrier) const {
  RECORD_FUNCTION("launch::H2D", {});
  resolveRuntimeHandle()->copy_h2d(src, src_capacity, dst, nullptr, barrier);
}

void SpyreStream::copyD2H(spyre_rt::DeviceView src, uint8_t* dst,
                          spyre_rt::ByteSize dst_capacity,
                          spyre_rt::PipelineBarrier barrier) const {
  RECORD_FUNCTION("launch::D2H", {});
  resolveRuntimeHandle()->copy_d2h(src, dst, dst_capacity, nullptr, barrier);
}

void SpyreStream::launchComputeStep(
    spyre_rt::DeviceView program, spyre_rt::BootstrapOffset bootstrap,
    rust::Slice<const spyre_rt::DeviceView> tensors,
    spyre_rt::PipelineBarrier barrier) const {
  RECORD_FUNCTION("launch::Compute", {});
  resolveRuntimeHandle()->launch_compute(program, bootstrap, tensors, barrier);
}

void SpyreStream::uploadCorrectionsStep(
    rust::Box<spyre_rt::Corrections> corrections, spyre_rt::DeviceView dst,
    spyre_rt::PipelineBarrier barrier) const {
  RECORD_FUNCTION("launch::H2D", {});
  resolveRuntimeHandle()->upload_corrections(std::move(corrections), dst,
                                             barrier);
}

void SpyreStream::fill(spyre_rt::DeviceView dst, double value,
                       spyre_rt::DataFormat format) const {
  RECORD_FUNCTION("launch::Memset", {});
  resolveRuntimeHandle()->fill(dst, value, format);
}

void SpyreStream::launch(const JobPlan& plan,
                         const std::vector<at::Tensor>& args,
                         std::vector<SymbolicArg> symbolic_args) const {
  // Validate all tensors are on Spyre device
  for (size_t i = 0; i < args.size(); ++i) {
    TORCH_CHECK(args[i].is_privateuseone(), "SpyreStream::launch: argument ", i,
                " must be on Spyre device, got ", args[i].device());
  }

  // Two-stream overlap topology:
  //   S_dev  = this stream (the default) — Compute (+ D2H).
  //   S_prep = the persistent host-compute stream — HostCompute + separate H2D.
  // Correction uploads travel with their compute on S_dev (PF never batches:
  // maxProgramCorrections() is the vendor base default 0,
  // runtime_scheduler.hpp:141). HostCompute runs inline on the caller; only
  // uploads and compute are queued. S_dev compute can overlap preparation for
  // subsequent launches. Every op keeps pipeline_barrier=true (per-stream
  // FIFO). S_prep must be the same persistent runtime stream each launch:
  // getHostComputeStreamById is a pure lookup of the handle registered once
  // in initializeStreamPoolImpl.
  const SpyreStream& s_dev = *this;
  const SpyreStream s_prep =
      getHostComputeStreamById(kHostComputeStreamStartPerDevice, device());

  // Create launch context with tensor arguments and typed symbolic payload.
  // symbolic_args is moved in so JobPlanStepHostCompute::construct can
  // consume it without an extra copy.
  LaunchContext ctx{args, std::move(symbolic_args)};
  const size_t max_corrections =
      s_dev.resolveRuntimeHandle()->max_program_corrections();

  // Split Prep-role steps onto S_prep only when the hazard tracker is on;
  // the runtime then inserts the cross-stream edges. Off = every step on
  // S_dev (the single-stream floor). Routing keys on role(), so all-Dev
  // plans never split. This switch controls stream splitting only: eligible
  // correction batching is enabled in both topologies, including the
  // single-stream path.
  const bool should_split = get_hazard_tracker_enabled();
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    const auto& step = plan.steps[i];
    const auto kind = classifyStep(*step);
    // Group by operation kind: Prep also includes standalone H2D steps, which
    // must not be collected as host corrections. Pin the role assumptions
    // used by the routing below so new step variants cannot silently change
    // them.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(kind != StepKind::HostCompute ||
                                     step->role() == StreamRole::Prep);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(kind != StepKind::Compute ||
                                     step->role() == StreamRole::Dev);
    if (max_corrections > 0 && kind == StepKind::HostCompute &&
        (i == 0 || classifyStep(*plan.steps[i - 1]) != StepKind::HostCompute)) {
      size_t end = i + 1;
      while (end < plan.steps.size() &&
             classifyStep(*plan.steps[end]) == StepKind::HostCompute) {
        ++end;
      }
      // The runtime reports the supported correction count for the consuming
      // stream. Keep the whole group on separate uploads when it cannot be
      // batched.
      ctx.batch_program_corrections =
          end - i <= max_corrections && end < plan.steps.size() &&
          classifyStep(*plan.steps[end]) == StepKind::Compute;
    }
    const SpyreStream& target =
        (should_split && step->role() == StreamRole::Prep) ? s_prep : s_dev;
    step->construct(ctx, target);
  }
#ifdef TORCH_SPYRE_FLEX
  TORCH_CHECK(ctx.program_corrections.empty(),
              "Program correction has no consuming compute step");
#endif
}

void initializeStreamPoolImpl(c10::DeviceIndex device_index) {
  auto& pool = getStreamPool();
  std::unique_lock<std::shared_mutex> lock(pool.mutex);

  // Check that this is the first and only device initialization
  TORCH_CHECK(!pool.initialized,
              "initializeStreamPoolImpl already called with device_index ",
              static_cast<int>(pool.initialized_device_index),
              "; cannot reinitialize with device_index ",
              static_cast<int>(device_index));
  pool.initialized = true;
  pool.initialized_device_index = device_index;

  // Initialize mapping from StreamId → runtime Stream.
  // The Stream instances are owned by GlobalRuntime.
  // StreamPool only stores non-owning pointers for lookup.
  auto runtime = GlobalRuntime::get();
  TORCH_CHECK(runtime != nullptr,
              "initializeStreamPoolImpl: the spyre_rt runtime is not "
              "initialized (was start_runtime called?)");

  // Register default stream (ID 0).
  pool.stream_handle_map[0] = &runtime->default_stream();

  // Register host compute streams (IDs 65+)
  int num_host_streams = getNumHostComputeStreams();
  pool.host_compute_streams[device_index].reserve(num_host_streams);
  for (int i = 0; i < num_host_streams; ++i) {
    c10::StreamId sid = kHostComputeStreamStartPerDevice + i;
    // Create the runtime stream only if this stream id has not been
    // registered yet. stream_handle_map is keyed by stream id, not device, so
    // when the pool is initialized under more than one device index in the
    // same process (all mapping to the same runtime) the handle is created
    // once and reused.
    TORCH_CHECK(
        pool.stream_handle_map.find(sid) == pool.stream_handle_map.end(),
        "Host compute stream id ", sid,
        " is already registered; only one Spyre device per process is "
        "supported.");
    pool.stream_handle_map[sid] = &runtime->create_stream(
        spyre_rt::StreamPriority::Normal, spyre_rt::StreamOrdering::Strict,
        /*track_hazards=*/get_hazard_tracker_enabled()
            ? spyre_rt::HazardTracking::Tracked
            : spyre_rt::HazardTracking::Untracked);
    pool.host_compute_streams[device_index].push_back(sid);
  }
  pool.next_host_compute_idx[device_index] = 0;

  // Initialize low priority streams (IDs 1 to kStreamsPerDevice)
  pool.low_priority_streams[device_index].reserve(kStreamsPerDevice);
  for (int i = 1; i <= kStreamsPerDevice; ++i) {
    pool.low_priority_streams[device_index].push_back(i);
  }
  pool.next_low_priority_idx[device_index] = 0;

  pool.high_priority_streams[device_index].reserve(
      kHighPriorityStreamsPerDevice);
  for (int i = 1; i <= kHighPriorityStreamsPerDevice; ++i) {
    pool.high_priority_streams[device_index].push_back(kStreamsPerDevice + i);
  }
  pool.next_high_priority_idx[device_index] = 0;
}

void initializeStreamPool(c10::DeviceIndex device_index) {
  auto& pool = getStreamPool();
  std::call_once(pool.device_init_flags[device_index], initializeStreamPoolImpl,
                 device_index);
}

SpyreStream getDefaultStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }
  initializeStreamPool(device.index());
  return SpyreStream(c10::Stream(c10::Stream::DEFAULT, device));
}

const spyre_rt::Stream* getDefaultStreamRuntimeHandle(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }
  initializeStreamPool(device.index());

  auto& pool = getStreamPool();
  std::shared_lock<std::shared_mutex> lock(pool.mutex);
  auto it = pool.stream_handle_map.find(0);
  TORCH_CHECK(it != pool.stream_handle_map.end(),
              "Default stream handle not initialized for device ",
              device.index());
  return it->second;
}

SpyreStream getCurrentStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  auto it = current_streams.find(device.index());
  if (it == current_streams.end()) {
    return getDefaultStream(device);
  }

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, it->second));
}

SpyreStream setCurrentStream(SpyreStream stream) {
  auto device = stream.device();
  auto old_stream = getCurrentStream(device);
  current_streams[device.index()] = stream.id();
  return old_stream;
}

SpyreStream getHostComputeStream(c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  // Ensure runtime is initialized before creating streams
  // This is critical when this is called before any tensor operations
  startRuntime();

  initializeStreamPool(device.index());

  auto& pool = getStreamPool();
  std::unique_lock<std::shared_mutex> lock(pool.mutex);

  auto& streams = pool.host_compute_streams[device.index()];
  auto& idx = pool.next_host_compute_idx[device.index()];

  c10::StreamId stream_id = streams[idx];
  idx = (idx + 1) % streams.size();

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, stream_id));
}

SpyreStream getHostComputeStreamById(c10::StreamId id, c10::Device device) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  // Ensure runtime is initialized before creating streams
  // This is critical when this is called before any tensor operations
  startRuntime();

  initializeStreamPool(device.index());

  const c10::StreamId end =
      kHostComputeStreamStartPerDevice + getNumHostComputeStreams();
  TORCH_CHECK(id >= kHostComputeStreamStartPerDevice && id < end,
              "getHostComputeStreamById: stream id ", id,
              " is not a host compute stream id (valid range [",
              kHostComputeStreamStartPerDevice, ", ", end, "))");

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, id));
}

SpyreStream getStreamFromPool(c10::Device device, int priority) {
  if (device.index() == -1) {
    device = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
  }

  // Ensure runtime is initialized before creating streams
  // This is critical when torch.Stream() is called before any tensor operations
  startRuntime();

  initializeStreamPool(device.index());

  auto& pool = getStreamPool();
  std::unique_lock<std::shared_mutex> lock(pool.mutex);

  c10::StreamId stream_id;
  if (priority == 0) {
    // Low priority stream - round-robin from low priority pool
    auto& streams = pool.low_priority_streams[device.index()];
    auto& idx = pool.next_low_priority_idx[device.index()];

    stream_id = streams[idx];
    idx = (idx + 1) % streams.size();

  } else {
    // High priority stream - round-robin from high priority pool
    auto& streams = pool.high_priority_streams[device.index()];
    auto& idx = pool.next_high_priority_idx[device.index()];

    stream_id = streams[idx];
    idx = (idx + 1) % streams.size();
  }

  // Create corresponding runtime stream (if not exists)
  if (pool.stream_handle_map.find(stream_id) == pool.stream_handle_map.end()) {
    auto runtime = GlobalRuntime::get();
    spyre_rt::StreamPriority streamPriority =
        priority < 0 ? spyre_rt::StreamPriority::High
                     : spyre_rt::StreamPriority::Normal;
    const spyre_rt::Stream* handle = &runtime->create_stream(
        streamPriority, spyre_rt::StreamOrdering::Strict,
        /*track_hazards=*/get_hazard_tracker_enabled()
            ? spyre_rt::HazardTracking::Tracked
            : spyre_rt::HazardTracking::Untracked);
    pool.stream_handle_map[stream_id] = handle;
  }

  return SpyreStream(c10::Stream(c10::Stream::UNSAFE, device, stream_id));
}

void synchronizeDevice(c10::optional<c10::Device> device) {
  auto sync_one_device = [](c10::Device dev) {
    if (dev.index() == -1) {
      dev = c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx);
    }
    const auto device_index = dev.index();

    std::vector<const spyre_rt::Stream*> handles_to_sync;
    {
      auto& pool = getStreamPool();
      std::shared_lock<std::shared_mutex> lock(pool.mutex);

      // Default stream (ID 0) is always present when the pool is initialized
      auto default_it = pool.stream_handle_map.find(0);
      if (default_it != pool.stream_handle_map.end()) {
        handles_to_sync.push_back(default_it->second);
      }

      auto collect = [&](auto& stream_map) {
        auto it = stream_map.find(device_index);
        if (it == stream_map.end()) return;
        for (auto sid : it->second) {
          auto h = pool.stream_handle_map.find(sid);
          if (h != pool.stream_handle_map.end()) {
            handles_to_sync.push_back(h->second);
          }
        }
      };
      collect(pool.low_priority_streams);
      collect(pool.high_priority_streams);
      collect(pool.host_compute_streams);
    }  // lock released

    c10::DeviceGuard guard(dev);
    for (auto handle : handles_to_sync) {
      handle->synchronize();
    }
  };
  if (device.has_value()) {
    sync_one_device(device.value());
  } else {
    sync_one_device(
        c10::Device(c10::DeviceType::PrivateUse1, SpyreGuardImpl::tls_idx));
  }
}

const char* SpyreStreamGetErrorString(SpyreStreamError error) noexcept {
  switch (error) {
    case SpyreStreamError::Success:
      return "Success";
    case SpyreStreamError::Shutdown:
      return "Shutdown";
    default:
      return "Unknown";
  }
}

SpyreStreamError SpyreStreamGetError(const SpyreStream& stream) {
  return stream.getError();
}

SpyreDeviceState SpyreGetDeviceState() {
  auto runtime = GlobalRuntime::get();
  if (!runtime) {
    return SpyreDeviceState::NotInitialized;
  }
  // NOTE: intentionally uses the runtime's aggregate has_stream_error() (one
  // locked iteration) instead of rolling up SpyreStreamGetError per stream.
  // These agree only while both reduce to the shutdown state.
  // TODO(#3365): once the runtime exposes typed per-stream codes, roll up via
  // SpyreStreamGetError so the aggregate (and the pytest skip reason) can
  // surface the actual fault class.
  if (runtime->has_stream_error()) {
    return SpyreDeviceState::StreamError;
  }
  return SpyreDeviceState::Ok;
}

}  // namespace spyre
