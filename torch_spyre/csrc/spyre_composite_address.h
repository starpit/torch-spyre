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

#pragma once

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "spyre_rt.h"

namespace spyre {

// Per-chunk geometry of a device allocation, copied out of a
// spyre_rt::DeviceView for read-only inspection from Python. Kept under
// TORCH_SPYRE_FLEX with the handle below (tests/debug only in the PoC).
struct CompositeChunkInfo {
  uint64_t region_id;
  uint64_t offset;
  size_t size;
  uint32_t domain_id;

  bool operator==(const CompositeChunkInfo& o) const {
    return region_id == o.region_id && offset == o.offset && size == o.size &&
           domain_id == o.domain_id;
  }
  bool operator!=(const CompositeChunkInfo& o) const {
    return !(*this == o);
  }
};

// Raw-pointer accessor for C++ callers: returns a non-owning pointer to the
// spyre_rt::DeviceView cached in `tensor`'s SharedOwnerCtx. Valid only while
// the tensor's storage stays alive -- caller must keep the tensor alive for
// as long as the returned pointer is used.
//
// Only storage-level invariants are validated (on-device, non-null storage
// and context) so that C++ callers that legitimately operate on
// non-contiguous views -- for example D2H copies of a transposed view whose
// layout is described separately via DataConversionInfo -- can still reach
// the view backing their view's storage.
spyre_rt::DeviceView* get_composite_address(const at::Tensor& tensor);

#ifdef TORCH_SPYRE_FLEX

// An opaque, read-only handle over the device view that backs a
// device("spyre") tensor's storage (tests/debug only; the PoC build does not
// define TORCH_SPYRE_FLEX). The handle keeps the source tensor alive by
// holding its storage and exposes the allocation's chunk geometry.
class CompositeAddressHandle {
 public:
  explicit CompositeAddressHandle(const at::Tensor& tensor);

  size_t total_size() const;
  size_t num_chunks() const;
  std::vector<CompositeChunkInfo> chunks() const;

 private:
  c10::Storage storage_;
  spyre_rt::DeviceView view_;
};

// Python-facing accessor (tests/debug only in the PoC).
CompositeAddressHandle get_composite_address_handle(const at::Tensor& tensor);

#endif  // TORCH_SPYRE_FLEX

}  // namespace spyre
