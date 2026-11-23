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

// The spyre_rt build's collectives: card to card over the senlib-rs bridge's
// peer link (spyre-torch `tp_*`, flex-rs `peer_link`), with no collective
// library. vLLM runs one process per rank, each with its own card:
//  - The backend's constructor links this rank's card once. It publishes the
//    card's PCI address through the group's Store, reads every rank's,
//    links (tp_link), then publishes its landing window and installs every
//    peer's.
//  - all-reduce (sum): send this rank's tensor into every peer's slot for it,
//    wait for every peer's, then sum on device IN RANK ORDER (part 0 + part 1
//    + ...). Every rank adds in the same order, so every rank holds the same
//    bits.
//  - all-gather: the same send and wait, then the parts concatenated in rank
//    order.
// The eager c10d backend (spyreccl) and the compiled path's plan/run ops
// (spyre::allreduce_plan / allreduce_run, ...) share these two bodies.
// Broadcast and reduce-scatter are refused by name: spyre-inference's
// tensor parallelism reaches neither on the device group.

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/library.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../module.h"
#include "../spyre_allocator.h"
#include "../spyre_composite_address.h"
#include "../spyre_rt_bridge.h"
#include "../spyre_storage_impl.h"
#include "../spyre_stream.h"
#include "../spyre_tensor_impl.h"

namespace spyre::rt_ccl {

// One landing slot: the most device bytes one collective sends one peer.
// The window is 2 parities x world slots of it on every card.
constexpr uint64_t kSlotBytes = uint64_t{16} << 20;

// This process's card-to-card world: one link per process. vLLM makes several
// process groups on this backend (the default group, then its world, tensor-
// and pipeline-parallel groups). A group of one rank needs no link; every
// other group must be the linked world itself, the same cards in the same rank
// order, and shares its link and its parity count (every rank calls the
// groups' collectives in one order).
struct World {
  std::mutex mu;
  int rank = -1;
  int size = 0;
  // The world's cards, in rank order, as the link was made.
  std::vector<std::string> pcis;
  // Collectives so far: collective n lands at parity n % 2.
  uint64_t collectives = 0;
};

World& world() {
  static World w;
  return w;
}

spyre_rt::Runtime& runtime() {
  auto* rt = spyre::GlobalRuntime::get();
  TORCH_CHECK(rt != nullptr,
              "spyreccl: the Spyre runtime is not started (torch.spyre's "
              "_lazy_init) before the process group");
  return *rt;
}

std::vector<uint8_t> bytes_of(const std::string& s) {
  return std::vector<uint8_t>(s.begin(), s.end());
}

std::string key(const char* what, int rank) {
  return std::string("spyre_rt_tp/") + what + "/" + std::to_string(rank);
}

// Links this rank's card for the group behind `store` (of two ranks or more):
// the first such group makes the link; a later one must be the linked world.
void link(c10d::Store& store, int rank, int size) {
  auto& w = world();
  std::lock_guard<std::mutex> lock(w.mu);
  TORCH_CHECK(size >= 2 && size <= 64, "spyreccl: a group of ", size,
              " ranks; the card-to-card link holds 2..64");
  auto& rt = runtime();
  // 1. Every rank's card, in rank order, through this group's own Store.
  store.set(key("pci", rank), bytes_of(std::string(rt.card_pci())));
  std::vector<std::string> pcis(static_cast<size_t>(size));
  for (int r = 0; r < size; ++r) {
    auto v = store.get(key("pci", r));
    pcis[static_cast<size_t>(r)] = std::string(v.begin(), v.end());
  }
  if (w.size != 0) {
    TORCH_CHECK(rank == w.rank && pcis == w.pcis,
                "spyreccl: a group of ", size, " ranks (this rank ", rank,
                ") that is not the linked world of ", w.size, " (rank ",
                w.rank,
                "); the card-to-card link is one per process, and every group "
                "of two ranks or more must be that world");
    return;
  }
  // 2. Link this card, then publish where its window sits. A peer sends only
  //    after it reads this window, which is published after the landing is
  //    programmed.
  spyre_rt::WindowWords at = spyre_rt::vendor([&](spyre_rt::Thrown& t) {
    return rt.tp_link(t, static_cast<uint8_t>(rank), pcis,
                      spyre_rt::ByteSize{kSlotBytes});
  });
  std::vector<uint8_t> words(16);
  std::memcpy(words.data(), &at.segment, 8);
  std::memcpy(words.data() + 8, &at.offset_bytes, 8);
  store.set(key("window", rank), words);
  // 3. Every peer's window.
  for (int r = 0; r < size; ++r) {
    if (r == rank) {
      continue;
    }
    auto v = store.get(key("window", r));
    TORCH_CHECK(v.size() == 16, "spyreccl: rank ", r, "'s window is ",
                v.size(), " bytes, not 16");
    spyre_rt::WindowWords peer{};
    std::memcpy(&peer.segment, v.data(), 8);
    std::memcpy(&peer.offset_bytes, v.data() + 8, 8);
    spyre_rt::vendor([&](spyre_rt::Thrown& t) {
      rt.tp_set_peer_window(t, static_cast<uint8_t>(r), peer);
    });
  }
  w.rank = rank;
  w.size = size;
  w.pcis = std::move(pcis);
}

void delete_borrowed_owner(void* owner_ctx) {
  delete static_cast<SharedOwnerCtx*>(owner_ctx);
}

// The peer `from`'s message at `parity`, as a tensor shaped and laid out as
// `like`. It borrows the slot: no allocation, and the window outlives it.
at::Tensor slot_tensor(int from, uint8_t parity, const at::Tensor& like,
                       uint64_t device_bytes) {
  spyre_rt::DeviceView slot = spyre_rt::vendor([&](spyre_rt::Thrown& t) {
    return runtime().tp_slot(t, static_cast<uint8_t>(from), parity);
  });
  slot.size = spyre_rt::ByteSize{device_bytes};
  auto* owner = new SharedOwnerCtx(slot, like.device().index(),
                                   spyre_rt::MemoryType::Tensor);
  void* ctx = static_cast<void*>(owner);
  c10::DataPtr ptr(ctx, ctx, &delete_borrowed_owner, like.device());
  auto storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), static_cast<int64_t>(device_bytes),
      std::move(ptr), &SpyreAllocator::instance());
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto t = at::detail::make_tensor_base<SpyreTensorImpl>(
      c10::Storage(std::move(storage_impl)), pu1_dks, like.dtype());
  auto* impl = static_cast<SpyreTensorImpl*>(t.unsafeGetTensorImpl());
  const auto* like_impl =
      static_cast<const SpyreTensorImpl*>(like.unsafeGetTensorImpl());
  impl->set_sizes_and_strides(like.sizes(), like.strides());
  impl->spyre_layout = like_impl->spyre_layout;
  impl->dma_sizes = like_impl->dma_sizes;
  impl->dma_strides = like_impl->dma_strides;
  return at::Tensor(std::move(t));
}

// The checks both collectives make, and the whole-storage view this rank
// sends: a collective sends the tensor's allocation, so the tensor must be
// all of it.
spyre_rt::DeviceView whole_storage(const at::Tensor& t, const char* op) {
  TORCH_CHECK(t.is_privateuseone(), "spyreccl ", op,
              ": the tensor must be on the Spyre device, got ", t.device());
  TORCH_CHECK(t.is_contiguous() && t.storage_offset() == 0, "spyreccl ", op,
              ": the tensor must be contiguous at storage offset 0 (it sends "
              "its whole allocation)");
  TORCH_CHECK(t.numel() > 0, "spyreccl ", op, ": an empty tensor");
  TORCH_CHECK(dynamic_cast<const SpyreTensorImpl*>(t.unsafeGetTensorImpl()),
              "spyreccl ", op, ": the tensor carries no Spyre layout");
  const spyre_rt::DeviceView* view = get_composite_address(t);
  TORCH_CHECK(view != nullptr, "spyreccl ", op,
              ": the tensor has no device allocation");
  TORCH_CHECK(view->size.raw <= kSlotBytes, "spyreccl ", op, ": ",
              view->size.raw, " device bytes past the ", kSlotBytes,
              "-byte landing slot");
  return *view;
}

// Sends `t` to every peer at the next parity and waits for every peer's.
// Returns the parity, for reading the slots.
uint8_t exchange(const at::Tensor& t, spyre_rt::DeviceView src) {
  auto& w = world();
  TORCH_CHECK(w.size >= 2,
              "spyreccl: a collective before the process group's backend "
              "linked this card");
  const auto parity = static_cast<uint8_t>(w.collectives++ % 2);
  auto stream = getCurrentStream(t.device());
  auto& rt = runtime();
  for (int peer = 0; peer < w.size; ++peer) {
    if (peer != w.rank) {
      stream.tpSend(rt, static_cast<uint8_t>(peer), parity, src);
    }
  }
  stream.tpWait(rt, parity);
  return parity;
}

const at::Tensor& all_reduce_sum(const at::Tensor& input) {
  const auto src = whole_storage(input, "all_reduce");
  const auto parity = exchange(input, src);
  auto& w = world();
  at::Tensor acc;
  for (int r = 0; r < w.size; ++r) {
    at::Tensor part =
        r == w.rank ? input : slot_tensor(r, parity, input, src.size.raw);
    acc = acc.defined() ? at::add(acc, part) : part;
  }
  input.copy_(acc);
  return input;
}

at::Tensor all_gather(const at::Tensor& input) {
  const auto src = whole_storage(input, "all_gather");
  const auto parity = exchange(input, src);
  auto& w = world();
  std::vector<at::Tensor> parts;
  parts.reserve(static_cast<size_t>(w.size));
  for (int r = 0; r < w.size; ++r) {
    parts.push_back(r == w.rank ? input
                                : slot_tensor(r, parity, input, src.size.raw));
  }
  return at::cat(parts, 0);
}

[[noreturn]] void refuse(const char* op) {
  TORCH_CHECK(false, "spyreccl ", op,
              " is not available on the spyre_rt runtime: its collectives "
              "are all_reduce (sum) and all_gather");
}

// ---- the compiled path's ops (the same schemas as spyre_distributed.cpp) --

int64_t allreduce_plan(int64_t, int64_t, const std::string& reduce_op,
                       const std::string&) {
  TORCH_CHECK(reduce_op == "sum", "spyreccl all_reduce: only sum, got ",
              reduce_op);
  return 0;
}
int64_t allgather_plan(int64_t, int64_t, int64_t, const std::string&) {
  return 0;
}
int64_t broadcast_plan(int64_t, int64_t, int64_t, const std::string&) {
  refuse("broadcast");
}
int64_t reducescatter_plan(int64_t, int64_t, int64_t, const std::string&,
                           const std::string&) {
  refuse("reduce_scatter");
}
at::Tensor allreduce_run(const at::Tensor& input, int64_t) {
  return all_reduce_sum(input);
}
at::Tensor allgather_run(const at::Tensor& input, int64_t, int64_t group_size) {
  TORCH_CHECK(group_size == world().size, "spyreccl all_gather: group_size ",
              group_size, " in a world of ", world().size);
  return all_gather(input);
}
at::Tensor broadcast_run(const at::Tensor&, int64_t, int64_t) {
  refuse("broadcast");
}
at::Tensor reducescatter_run(const at::Tensor&, int64_t, int64_t) {
  refuse("reduce_scatter");
}
// The collectives are ordered on the stream, so there is nothing to wait for.
at::Tensor wait_work(const at::Tensor& tensor) {
  return tensor;
}

// ---- the compiled path's in-graph all-reduce ------------------------------
// The compiled graph sums the parts itself, in one kernel of its own, with the
// device layouts it knows (an eager add or out= op writes the default layout,
// and a graph buffer's layout is not always that). Two ops serve it:
//  - tp_exchange_run(input): this rank's tensor into its OWN slot (a copy_,
//    exact) and into every peer's slot for it, then the wait. Every part of
//    the collective now sits in this card's window, so the graph reads part r
//    from the window for every r and is the same graph on every rank. The
//    parity is recorded against input's storage.
//  - tp_part_run(input, part): rank `part`'s message of input's last
//    exchange, borrowed from the window and laid out as input.

std::unordered_map<const c10::StorageImpl*, uint8_t>& exchanged() {
  static std::unordered_map<const c10::StorageImpl*, uint8_t> m;
  return m;
}

void tp_exchange_run(const at::Tensor& input, int64_t) {
  const auto src = whole_storage(input, "all_reduce");
  const auto parity = exchange(input, src);
  // No peer writes this rank's own slot, so it is filled after the exchange.
  auto& w = world();
  slot_tensor(w.rank, parity, input, src.size.raw).copy_(input);
  std::lock_guard<std::mutex> lock(w.mu);
  exchanged()[input.storage().unsafeGetStorageImpl()] = parity;
}

at::Tensor tp_part_run(const at::Tensor& input, int64_t, int64_t part) {
  auto& w = world();
  TORCH_CHECK(part >= 0 && part < w.size, "spyreccl: part ", part,
              " in a world of ", w.size);
  uint8_t parity = 0;
  {
    std::lock_guard<std::mutex> lock(w.mu);
    const auto it = exchanged().find(input.storage().unsafeGetStorageImpl());
    TORCH_CHECK(it != exchanged().end(),
                "spyreccl: a part of a tensor this rank never exchanged");
    parity = it->second;
  }
  return slot_tensor(static_cast<int>(part), parity, input,
                     get_composite_address(input)->size.raw);
}

}  // namespace spyre::rt_ccl

TORCH_LIBRARY(spyre, m) {
  m.def(
      "broadcast_async(Tensor input, int src_rank, str group_name) -> Tensor");
  m.def(
      "all_gather_async(Tensor input, SymInt group_size=1, "
      "str group_name=\"default\") -> Tensor");
  m.def(
      "all_reduce_async(Tensor(a!) input, str reduce_op=\"sum\", "
      "str group_name=\"default\") -> Tensor(a)");
  m.def("wait_work(Tensor(a!) tensor) -> Tensor(a)");
  m.def(
      "broadcast_plan(int num_elems, int dtype, int src_rank, "
      "str group_name) -> int",
      &spyre::rt_ccl::broadcast_plan);
  m.def(
      "allreduce_plan(int num_elems, int dtype, str reduce_op, "
      "str group_name) -> int",
      &spyre::rt_ccl::allreduce_plan);
  m.def(
      "allgather_plan(int num_elems, int dtype, int group_size, "
      "str group_name) -> int",
      &spyre::rt_ccl::allgather_plan);
  m.def(
      "reducescatter_plan(int num_elems, int dtype, int group_size, "
      "str reduce_op, str group_name) -> int",
      &spyre::rt_ccl::reducescatter_plan);
  m.def(
      "broadcast_run(Tensor input, int plan_handle, int src_rank) "
      "-> Tensor");
  m.def("allreduce_run(Tensor(a!) input, int plan_handle) -> Tensor(a)");
  m.def(
      "allgather_run(Tensor input, int plan_handle, int group_size) "
      "-> Tensor");
  m.def(
      "reducescatter_run(Tensor input, int plan_handle, int group_size) "
      "-> Tensor");
  m.def("tp_exchange_run(Tensor(a!) input, int plan_handle) -> ()");
  m.def("tp_part_run(Tensor input, int plan_handle, int part) -> Tensor");
}

TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("wait_work", &spyre::rt_ccl::wait_work);
  m.impl("broadcast_run", &spyre::rt_ccl::broadcast_run);
  m.impl("allreduce_run", &spyre::rt_ccl::allreduce_run);
  m.impl("allgather_run", &spyre::rt_ccl::allgather_run);
  m.impl("reducescatter_run", &spyre::rt_ccl::reducescatter_run);
  m.impl("tp_exchange_run", &spyre::rt_ccl::tp_exchange_run);
  m.impl("tp_part_run", &spyre::rt_ccl::tp_part_run);
}

namespace c10d {

// A collective already ordered on its stream: complete when returned.
class SpyreRtWork final : public Work {
 public:
  SpyreRtWork(OpType op, std::vector<at::Tensor> result)
      : Work(-1, op),
        future_(c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()))) {
    future_->markCompleted(c10::IValue(std::move(result)));
  }
  bool isCompleted() override {
    return true;
  }
  bool isSuccess() const override {
    return true;
  }
  bool wait(std::chrono::milliseconds) override {
    return true;
  }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    return future_;
  }

 private:
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

class SpyreRtCCLBackend final : public Backend {
 public:
  // A group of one rank (vLLM's pipeline-parallel group at pp=1, say) needs
  // no link: its collectives are the identity.
  SpyreRtCCLBackend(const c10::intrusive_ptr<Store>& store, int rank, int size)
      : Backend(rank, size) {
    if (size > 1) {
      spyre::rt_ccl::link(*store, rank, size);
    }
  }

  const std::string getBackendName() const override {
    return "spyreccl";
  }

  c10::intrusive_ptr<Work> allreduce(std::vector<at::Tensor>& tensors,
                                     const AllreduceOptions& opts) override {
    TORCH_CHECK(tensors.size() == 1, "spyreccl all_reduce: one tensor, got ",
                tensors.size());
    TORCH_CHECK(opts.reduceOp == ReduceOp::SUM,
                "spyreccl all_reduce: only sum");
    if (getSize() > 1) {
      spyre::rt_ccl::all_reduce_sum(tensors[0]);
    }
    return c10::make_intrusive<SpyreRtWork>(OpType::ALLREDUCE, tensors);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs, const AllgatherOptions&) override {
    TORCH_CHECK(inputs.size() == 1 && outputs.size() == 1,
                "spyreccl all_gather: one input and one output list");
    TORCH_CHECK(static_cast<int>(outputs[0].size()) == getSize(),
                "spyreccl all_gather: ", outputs[0].size(),
                " outputs in a world of ", getSize());
    if (getSize() == 1) {
      outputs[0][0].copy_(inputs[0]);
      return c10::make_intrusive<SpyreRtWork>(OpType::ALLGATHER, outputs[0]);
    }
    at::Tensor gathered = spyre::rt_ccl::all_gather(inputs[0]);
    auto parts = gathered.chunk(getSize(), 0);
    for (int r = 0; r < getSize(); ++r) {
      outputs[0][static_cast<size_t>(r)].copy_(parts[static_cast<size_t>(r)]);
    }
    return c10::make_intrusive<SpyreRtWork>(OpType::ALLGATHER, outputs[0]);
  }

  static c10::intrusive_ptr<Backend> createSpyreCCLBackend(
      const c10::intrusive_ptr<Store>& store, int rank, int size,
      const std::chrono::duration<float>&) {
    return c10::make_intrusive<SpyreRtCCLBackend>(store, rank, size);
  }
};

}  // namespace c10d

namespace spyre::rt_ccl {
// module.cpp's binding target in the spyre_rt build.
c10::intrusive_ptr<c10d::Backend> create_backend(
    const c10::intrusive_ptr<c10d::Store>& store, int rank, int size,
    const std::chrono::duration<float>& timeout) {
  return c10d::SpyreRtCCLBackend::createSpyreCCLBackend(store, rank, size,
                                                        timeout);
}
}  // namespace spyre::rt_ccl
