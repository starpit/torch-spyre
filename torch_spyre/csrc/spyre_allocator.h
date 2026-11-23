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

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Stream.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "spyre_rt_bridge.h"

namespace spyre {

// Defined in spyre_mem.cpp: a single-shard view whose DataPtr context is the
// SharedOwnerCtx below. Forward-declared so the ctx can point back at it.
struct ShardViewCtx;

// Sharding of a tensor's storage: shard i (the i-th equal slice along
// shard_dim) lives in that shard's own allocation, encoded in the
// shard-local device layout (shard_device_bytes). Address-free.
struct ShardedStorageInfo {
  int64_t shard_dim;
  uint32_t num_shards;
  uint64_t shard_device_bytes;
  // Shape and dtype the storage was allocated for; the copy path checks
  // tensor metadata against them. Filled in by spyre_empty_sharded; the
  // dtype default makes a future caller that forgets to set it fail the
  // copy check instead of reading an indeterminate value.
  std::vector<int64_t> global_shape;
  c10::ScalarType dtype = c10::ScalarType::Undefined;
};

struct SharedOwnerCtx {
  // The owning device allocation, when this ctx owns one: non-sharded
  // storage's single allocation. Empty for sharded storage (the owning
  // per-shard allocations live in shards) and for a ShardViewCtx's owner
  // (the parent storage keeps the memory alive).
  std::optional<rust::Box<spyre_rt::Allocation>> allocation;
  // Cached view of `allocation` (or, for a shard view, of the parent's shard
  // chunk). Read through get_composite_address(); valid while the storage
  // that owns this ctx is alive.
  spyre_rt::DeviceView view;
  // Sharded storage only: one owning allocation per shard, indexed by shard
  // ordinal (shard i backs core i), with its cached view in shard_views.
  // Empty for non-sharded storage.
  std::vector<rust::Box<spyre_rt::Allocation>> shard_allocations;
  std::vector<spyre_rt::DeviceView> shard_views;
  signed char device_id;
  spyre_rt::MemoryType memory_type;
  // Set only for sharded storage.
  std::optional<ShardedStorageInfo> sharding;
  // Set only on the SharedOwnerCtx embedded in a ShardViewCtx: points back
  // at the enclosing view so the view's DataPtr deleter can free it.
  ShardViewCtx* shard_view = nullptr;

  // Total physical device bytes: the whole allocation, or the sum over the
  // shard table. The vendor's CompositeAddress::total_size().
  uint64_t total_size() const {
    if (allocation.has_value()) {
      return view.size.raw;
    }
    uint64_t total = 0;
    for (const auto& v : shard_views) {
      total += v.size.raw;
    }
    return total;
  }

  SharedOwnerCtx(rust::Box<spyre_rt::Allocation> alloc, signed char dev_id,
                 spyre_rt::MemoryType mem_type)
      : allocation(std::move(alloc)),
        view(allocation.has_value() ? (*allocation)->view()
                                    : spyre_rt::DeviceView{}),
        device_id(dev_id),
        memory_type(mem_type) {}

  // For a shard view's non-owning owner: borrows the shard's view.
  SharedOwnerCtx(spyre_rt::DeviceView borrowed_view, signed char dev_id,
                 spyre_rt::MemoryType mem_type)
      : view(borrowed_view), device_id(dev_id), memory_type(mem_type) {}
};

// A custom allocator for our custom device, which returns a handle to the
// allocated memory, not the actual pointer.
struct SpyreAllocator final : public c10::DeviceAllocator {
 private:
  SpyreAllocator();
  static c10::CachingDeviceAllocator::DeviceStats stats_;
  static c10::CachingDeviceAllocator::StatTypes
      stat_types;  // {AGGREGATE, SMALL_POOL, LARGE_POOL}
  static std::mutex stats_mutex_;

 public:
  static SpyreAllocator& instance();
  bool initialized() override;

  void emptyCache(c10::MempoolId_t mempool_id) override;

  void recordStream(const c10::DataPtr& ptr, c10::Stream stream) override;

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;

  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override;

  void resetAccumulatedStats(c10::DeviceIndex device) override;

  void resetPeakStats(c10::DeviceIndex device) override;

  void recordAlloc(size_t nbytes, void* data, int device,
                   spyre_rt::MemoryType memory_type);

  void recordRelease(size_t nbytes, void* data, int device,
                     spyre_rt::MemoryType memory_type);

  c10::DataPtr allocate(size_t nbytes) override;

  // Typed allocation through spyre_rt: the vendor's AllocationDirective
  // (PlacementPolicy::Bind over an explicit domain list) reduced to its
  // (memory type, domains) content, which is all the csrc ever states.
  c10::DataPtr allocate(size_t nbytes, spyre_rt::MemoryType memory_type,
                        const std::vector<spyre_rt::DomainId>& domains);

  // Allocate sharded storage: num_shards plain per-shard allocations of
  // shard_device_bytes, shard i bound to memory domain
  // i % num_memory_domains, tagged with the sharding record. The returned
  // DataPtr owns them through a SharedOwnerCtx.
  static c10::DataPtr allocateShardedStorage(int64_t shard_dim,
                                             uint32_t logical_chunks,
                                             uint64_t shard_device_bytes);

  static void ReportAndDelete(void* ctx_void);

  c10::DeleterFnPtr raw_deleter() const override;

  void copy_data(void* dest, const void* src, std::size_t count) const final;

#ifdef TORCH_SPYRE_FLEX
  // The vendor's compositeAddressToDeviceAddress: the device-addressable
  // (segment) address of a view. Sharded path only; device_address is not
  // in the spyre_rt bridge, so the whole translation stays under the ifdef.
  uint64_t compositeAddressToDeviceAddress(
      const spyre_rt::DeviceView& view) const;
#endif  // TORCH_SPYRE_FLEX
};

}  // namespace spyre
