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
#include "spyre_allocator.h"

#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_mem.h"
#include "spyre_tensor_impl.h"

namespace spyre {

SpyreAllocator::SpyreAllocator() = default;

c10::CachingDeviceAllocator::DeviceStats SpyreAllocator::stats_;
c10::CachingDeviceAllocator::StatTypes SpyreAllocator::stat_types = {
    true, false, false};  // {AGGREGATE, SMALL_POOL, LARGE_POOL}
std::mutex SpyreAllocator::stats_mutex_;

SpyreAllocator& SpyreAllocator::instance() {
  static SpyreAllocator allocator;
  return allocator;
}

bool SpyreAllocator::initialized() {
  return true;
}

void SpyreAllocator::emptyCache(c10::MempoolId_t mempool_id) {}

void SpyreAllocator::recordStream(const c10::DataPtr& ptr, c10::Stream stream) {
}

c10::CachingDeviceAllocator::DeviceStats SpyreAllocator::getDeviceStats(
    c10::DeviceIndex device) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return stats_;
}

std::pair<size_t, size_t> SpyreAllocator::getMemoryInfo(
    c10::DeviceIndex device) {
  auto* runtime = GlobalRuntime::get();
  TORCH_CHECK(runtime != nullptr,
              "SpyreAllocator::getMemoryInfo: the spyre_rt runtime is not "
              "initialized");
  // The vendor checks the requested index against the runtime's active
  // device; spyre_rt owns one device per process, so the index must be 0
  // (or -1 resolved to it) exactly as before.
  spyre_rt::MemoryStats mem_stats =
      spyre_rt::vendor([&](spyre_rt::Thrown& t) {
        return runtime->memory_stats(t, spyre_rt::MemoryType::Tensor);
      });
  return {mem_stats.free.raw, mem_stats.total.raw};
}

void SpyreAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocated_bytes[stat_type].reset_accumulated();
        stats_.reserved_bytes[stat_type].reset_accumulated();
        stats_.allocation[stat_type].reset_accumulated();
      });
}

void SpyreAllocator::resetPeakStats(c10::DeviceIndex device) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  c10::CachingAllocator::for_each_selected_stat_type(
      stat_types, [&](size_t stat_type) {
        stats_.allocated_bytes[stat_type].reset_peak();
        stats_.reserved_bytes[stat_type].reset_peak();
        stats_.allocation[stat_type].reset_peak();
      });
}

void SpyreAllocator::recordAlloc(size_t nbytes, void* data, int device_id,
                                 spyre_rt::MemoryType memory_type) {
  // User-visible accounting (stats and profiler) is Tensor-only.
  // Program and other non-Tensor allocations must not appear in user-visible
  // memory stats: on release the ctx is already gone, so nbytes would be 0
  // and would leave an unmatched +N in the profiler.
  if (memory_type != spyre_rt::MemoryType::Tensor) {
    return;
  }
  int64_t total_allocated;
  int64_t total_reserved;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    c10::CachingAllocator::for_each_selected_stat_type(
        stat_types, [&](size_t stat_type) {
          stats_.allocation[stat_type].increase(1);
          stats_.allocated_bytes[stat_type].increase(nbytes);
          stats_.reserved_bytes[stat_type].increase(nbytes);
        });
    total_allocated = stats_
                          .allocated_bytes[static_cast<size_t>(
                              c10::CachingAllocator::StatType::AGGREGATE)]
                          .current;
    total_reserved = stats_
                         .reserved_bytes[static_cast<size_t>(
                             c10::CachingAllocator::StatType::AGGREGATE)]
                         .current;
  }
  c10::Device curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_id);
  c10::reportMemoryUsageToProfiler(data,
                                   nbytes,           // alloc_size
                                   total_allocated,  // total_allocated
                                   total_reserved,   // total_reserved
                                   curr_device);
}

void SpyreAllocator::recordRelease(size_t nbytes, void* data, int device_id,
                                   spyre_rt::MemoryType memory_type) {
  // User-visible accounting (stats and profiler) is Tensor-only.
  if (memory_type != spyre_rt::MemoryType::Tensor) {
    return;
  }
  int64_t total_allocated;
  int64_t total_reserved;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    c10::CachingAllocator::for_each_selected_stat_type(
        stat_types, [&](size_t stat_type) {
          stats_.allocation[stat_type].decrease(1);
          stats_.allocated_bytes[stat_type].decrease(nbytes);
          stats_.reserved_bytes[stat_type].decrease(nbytes);
        });
    total_allocated = stats_
                          .allocated_bytes[static_cast<size_t>(
                              c10::CachingAllocator::StatType::AGGREGATE)]
                          .current;
    total_reserved = stats_
                         .reserved_bytes[static_cast<size_t>(
                             c10::CachingAllocator::StatType::AGGREGATE)]
                         .current;
  }
  c10::Device curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, device_id);
  c10::reportMemoryUsageToProfiler(data,
                                   -static_cast<int64_t>(nbytes),  // alloc_size
                                   total_allocated,  // total_allocated
                                   total_reserved,   // total_reserved
                                   curr_device);
}

c10::DataPtr SpyreAllocator::allocate(size_t nbytes) {
  // The vendor's default directive: PlacementPolicy::Bind, domain {0},
  // MemoryType::Tensor.
  return SpyreAllocator::allocate(nbytes, spyre_rt::MemoryType::Tensor,
                                  {spyre_rt::DomainId{0}});
}

c10::DataPtr SpyreAllocator::allocate(
    size_t nbytes, spyre_rt::MemoryType memory_type,
    const std::vector<spyre_rt::DomainId>& domains) {
  c10::Device curr_device =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)->getDevice();

  auto device_id = curr_device.index();

  SPYRE_RUNTIME_DEBUG() << "allocating " << nbytes << " (bytes) on Spyre"
                        << curr_device;
  if (nbytes == 0) {
    return {nullptr, nullptr, &ReportAndDelete, curr_device};
  }
  auto* runtime = GlobalRuntime::get();
  TORCH_CHECK(runtime != nullptr,
              "SpyreAllocator::allocate: the spyre_rt runtime is not "
              "initialized (was start_runtime called?)");

  rust::Box<spyre_rt::Allocation> allocation =
      spyre_rt::vendor([&](spyre_rt::Thrown& t) {
        return runtime->allocate(
            t, spyre_rt::ByteSize{nbytes}, memory_type,
            rust::Slice<const spyre_rt::DomainId>(domains.data(),
                                                  domains.size()));
      });

  auto* ctx = new SharedOwnerCtx(std::move(allocation), device_id,
                                 memory_type);
  void* ctx_void = static_cast<void*>(ctx);

  // Use the SharedOwnerCtx pointer as the unique data handle for c10::DataPtr.
  // This pointer is never dereferenced — it serves only as a unique token for
  // memory profiling (recordAlloc/recordRelease).
  void* data_void = static_cast<void*>(ctx);
  // The runtime rounds up to the device alignment (the vendor's
  // DEVICE_ALIGNMENT, 128 bytes), so the actual allocation may be larger
  // than the requested nbytes. Use the cached view's size for accurate
  // memory profiling.
  recordAlloc(ctx->view.size.raw, data_void, device_id, memory_type);

  auto data_ptr_result =
      at::DataPtr(data_void, ctx_void, &ReportAndDelete, curr_device);

  return data_ptr_result;
}

c10::DataPtr SpyreAllocator::allocateShardedStorage(
    int64_t shard_dim, uint32_t logical_chunks, uint64_t shard_device_bytes) {
  TORCH_CHECK(shard_device_bytes != 0,
              "sharded storage requires nonzero shard bytes");
  TORCH_CHECK(logical_chunks == 8 || logical_chunks == 16,
              "sharded storage supports 8 or 16 shards, got ", logical_chunks);
  TORCH_CHECK(shard_device_bytes % 128 == 0,
              "sharded storage requires device-aligned shard bytes, got ",
              shard_device_bytes);
  TORCH_CHECK(
      shard_device_bytes <= std::numeric_limits<size_t>::max() / logical_chunks,
      "sharded storage size overflows");

  c10::Device curr_device =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)->getDevice();
  auto* runtime = GlobalRuntime::get();
  TORCH_CHECK(runtime != nullptr,
              "SpyreAllocator::allocateShardedStorage: the spyre_rt runtime "
              "is not initialized");

  const uint32_t num_domains = runtime->num_domains().raw;

  // Each shard is its own aligned plain allocation; shard i backs core i by
  // the torch-spyre-owned shard table, bound to Flex memory domain
  // i % num_domains from the topology (the vendor's allocateShardTable).
  std::vector<rust::Box<spyre_rt::Allocation>> shard_allocations;
  std::vector<spyre_rt::DeviceView> shard_views;
  shard_allocations.reserve(logical_chunks);
  shard_views.reserve(logical_chunks);
  for (uint32_t i = 0; i < logical_chunks; ++i) {
    std::vector<spyre_rt::DomainId> domains{
        spyre_rt::DomainId{i % num_domains}};
    shard_allocations.push_back(
        spyre_rt::vendor([&](spyre_rt::Thrown& t) {
          return runtime->allocate(
              t, spyre_rt::ByteSize{shard_device_bytes},
              spyre_rt::MemoryType::Tensor,
              rust::Slice<const spyre_rt::DomainId>(domains.data(),
                                                    domains.size()));
        }));
    shard_views.push_back(shard_allocations.back()->view());
    TORCH_CHECK(shard_views.back().size.raw == shard_device_bytes,
                "plain per-shard allocation ", i, " covers ",
                shard_views.back().size.raw, " bytes, expected ",
                shard_device_bytes);
  }

  const size_t actual_nbytes =
      static_cast<size_t>(shard_device_bytes) * logical_chunks;
  auto device_id = curr_device.index();
  auto* ctx = new SharedOwnerCtx(
      spyre_rt::DeviceView{}, device_id, spyre_rt::MemoryType::Tensor);
  ctx->shard_allocations = std::move(shard_allocations);
  ctx->shard_views = std::move(shard_views);
  ctx->sharding =
      ShardedStorageInfo{shard_dim, logical_chunks, shard_device_bytes};
  void* ctx_void = static_cast<void*>(ctx);
  SpyreAllocator::instance().recordAlloc(actual_nbytes, ctx_void, device_id,
                                         spyre_rt::MemoryType::Tensor);
  return c10::DataPtr(ctx_void, ctx_void, &ReportAndDelete, curr_device);
}

void SpyreAllocator::ReportAndDelete(void* ctx_void) {
  if (!ctx_void) {
    return;
  }
  auto* ctx = static_cast<SharedOwnerCtx*>(ctx_void);
  size_t nbytes = ctx->total_size();

  SpyreAllocator::instance().recordRelease(nbytes, static_cast<void*>(ctx),
                                           ctx->device_id, ctx->memory_type);
  delete ctx;  // drops the Box(es); the runtime frees the device blocks
}

// The raw deleter only gets passed the data ptr, no context, so it
// would not work right now. To implement this, we first need to
// create a runtime interface that can correctly free an allocation
// only based on the data ptr, without the allocation idx from the
// context
c10::DeleterFnPtr SpyreAllocator::raw_deleter() const {
  return nullptr;
}

void SpyreAllocator::copy_data(void* dest, const void* src,
                               std::size_t count) const {
  py::gil_scoped_acquire acquire;
  SPYRE_RUNTIME_DEBUG() << "called";
  // do nothing -- look into when this is called
}

// device_address is not in the spyre_rt bridge (sharded path only); the
// translation stays with the rest of the sharded surface under the ifdef.
#ifdef TORCH_SPYRE_FLEX
uint64_t SpyreAllocator::compositeAddressToDeviceAddress(
    const spyre_rt::DeviceView& view) const {
  auto* runtime = GlobalRuntime::get();
  TORCH_CHECK(runtime != nullptr,
              "compositeAddressToDeviceAddress: the spyre_rt runtime is not "
              "initialized");
  return runtime->device_address(view);
}
#endif  // TORCH_SPYRE_FLEX

}  // namespace spyre

// The vendor's FlexAllocator memory-pressure callback (flex_allocator.cpp
// allocateInDomain): the Rust allocator calls this WITHOUT its lock held;
// it releases nothing itself, runs the Python GC, and returns. Retry
// semantics stay in the Rust allocator. Declared by the crate's
// include/spyre_torch.h in namespace spyre_rt (the bridge's one call back
// into its consumer); the link fails by name without this definition.
namespace spyre_rt {
void on_memory_pressure() {
  SPYRE_RUNTIME_DEBUG() << "memory pressure: acquiring Python GIL for "
                           "garbage collection";
  PyGILState_STATE gstate = PyGILState_Ensure();
  Py_ssize_t collected = PyGC_Collect();
  if (collected >= 0) {
    SPYRE_RUNTIME_DEBUG() << "PyGC_Collect() completed, collected "
                          << collected << " objects";
  } else {
    SPYRE_RUNTIME_DEBUG() << "PyGC_Collect() returned error";
  }
  PyGILState_Release(gstate);
  SPYRE_RUNTIME_DEBUG() << "memory pressure callback complete";
}
}  // namespace spyre_rt

// Register our custom allocator
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1,
                  &spyre::SpyreAllocator::instance());
