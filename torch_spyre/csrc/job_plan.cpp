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

#include "job_plan.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "spyre_composite_address.h"
#include "spyre_mem.h"
#include "spyre_stream.h"

namespace spyre {

namespace {
// The step's pipeline_barrier_ flag mapped onto the bridge enum.
inline spyre_rt::PipelineBarrier step_barrier(bool enabled) {
  return enabled ? spyre_rt::PipelineBarrier::Yes : spyre_rt::PipelineBarrier::No;
}
}  // namespace

void JobPlanStepH2D::construct(LaunchContext&,
                               const SpyreStream& stream) const {
  // The vendor's plain createDmaParams (no host capacity): the JobPlan owns
  // the pinned host buffer and outlives the in-flight DMA, and the buffer
  // provably matches the device allocation, so src_capacity is the device
  // size (the vendor validators check dma_size <= capacity only).
  stream.copyH2D(static_cast<const uint8_t*>(host_address_),
                 device_address_.size, device_address_, step_barrier(pipeline_barrier_));
}

void JobPlanStepH2D::write(std::ostream& os) const {
  os << "  H2D (Host-to-Device)\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Device view: "
     << std::string(spyre_rt::to_string(device_address_)) << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepD2H::construct(LaunchContext& ctx,
                               const SpyreStream& stream) const {
  if (std::holds_alternative<spyre_rt::DeviceView>(device_address_)) {
    const auto& device_address =
        std::get<spyre_rt::DeviceView>(device_address_);
    stream.copyD2H(device_address, static_cast<uint8_t*>(host_address_),
                   spyre_rt::ByteSize{size_}, step_barrier(pipeline_barrier_));
  } else {
    const uint64_t device_ptr = std::get<Dmva>(device_address_).value;
    const spyre_rt::DecodedPointer decoded =
        spyre_rt::decode_device_pointer(device_ptr);
    const uint32_t segment_id = decoded.segment;
    const uint64_t segment_offset = decoded.offset.raw;
    TORCH_CHECK(segment_id < ctx.inputs_outputs.size(),
                "D2H tensor-segment lookup out of range: segment ", segment_id,
                " but only ", ctx.inputs_outputs.size(),
                " launch args were provided");
    const auto& tensor = ctx.inputs_outputs.at(segment_id);
    const auto& tensor_view = *get_composite_address(tensor);
    TORCH_CHECK(segment_offset + size_ <= tensor_view.size.raw,
                "D2H transfer out of bounds: offset ", segment_offset,
                " + size ", size_, " exceeds tensor allocation size ",
                tensor_view.size.raw);
    // Offset view over the tensor's allocation: the vendor's
    // LogicalAddress/Chunk arithmetic reduced to DeviceView arithmetic.
    spyre_rt::DeviceView offset_view{
        tensor_view.region,
        spyre_rt::ByteOffset{tensor_view.offset.raw + segment_offset},
        spyre_rt::ByteSize{size_}, tensor_view.domain};
    stream.copyD2H(offset_view, static_cast<uint8_t*>(host_address_),
                   spyre_rt::ByteSize{size_}, step_barrier(pipeline_barrier_));
  }
}

void JobPlanStepD2H::write(std::ostream& os) const {
  os << "  D2H (Device-to-Host)\n";
  if (std::holds_alternative<spyre_rt::DeviceView>(device_address_)) {
    os << "    Device view: "
       << std::string(spyre_rt::to_string(
              std::get<spyre_rt::DeviceView>(device_address_)))
       << "\n";
  } else {
    os << "    Device dmva: " << std::get<Dmva>(device_address_).value << "\n";
  }
  os << "    Host address: " << host_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepCompute::construct(LaunchContext& ctx,
                                   const SpyreStream& stream) const {
  // Reuse the launch context's buffer: a launch allocates nothing.
  auto& tensor_views = ctx.tensor_views;
  tensor_views.clear();
  if (bind_io_addresses_) {
    for (auto& tensor : ctx.inputs_outputs) {
      tensor_views.push_back(*get_composite_address(tensor));
    }
  }
  stream.launchComputeStep(
      program_address_, spyre_rt::BootstrapOffset{bootstrap_offset_},
      rust::Slice<const spyre_rt::DeviceView>(tensor_views.data(),
                                              tensor_views.size()),
      step_barrier(pipeline_barrier_));
}

void JobPlanStepCompute::write(std::ostream& os) const {
  os << "  Device Compute\n";
  os << "    Name: " << (name_.empty() ? "(unnamed)" : name_) << "\n";
  os << "    Program view: "
     << std::string(spyre_rt::to_string(program_address_)) << "\n";
  os << "    Bind I/O addresses: " << (bind_io_addresses_ ? "yes" : "no")
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::vector<spyre_rt::HostComputeArg> JobPlanStepHostCompute::resolveSymbolicArgs(
    const std::vector<at::Tensor>& tensors,
    const std::vector<SymbolicArg>& symbolic_args) {
  std::vector<spyre_rt::HostComputeArg> resolved(symbolic_args.size());
  for (size_t i = 0; i < symbolic_args.size(); ++i) {
    const SymbolicArg& arg = symbolic_args[i];
    TORCH_CHECK(arg.tensor_id >= 0 &&
                    static_cast<size_t>(arg.tensor_id) < tensors.size(),
                "symbolic_args tensor_id out of range");
    switch (arg.kind) {
      case SymbolicArgKind::kAddress:
        resolved[i] = spyre_rt::HostComputeArg{
            spyre_rt::HostComputeArgKind::Address,
            *get_composite_address(tensors[arg.tensor_id]), /*value=*/0};
        break;
      case SymbolicArgKind::kShardAddress:
        // dim_index = core (= shard), value = expected per-shard bytes,
        // num_shards = expected shard count; all checked against the tensor.
        resolved[i] = spyre_rt::HostComputeArg{
            spyre_rt::HostComputeArgKind::Value, spyre_rt::DeviceView{},
            static_cast<int64_t>(shard_device_address(tensors[arg.tensor_id],
                                                      arg.dim_index,
                                                      arg.num_shards,
                                                      arg.value))};
        break;
      case SymbolicArgKind::kDimension: {
        // A varying dimension behind a symbolic loop bound. The bundle
        // declares it as
        //   !sdscbundle.input_arg<index, granularity=G, max_value=M>
        // and the trip count is derived from it at dispatch, so the value
        // bound here is the ONLY place the runtime size enters the program.
        // Nothing in the program body is rewritten for it, which is the whole
        // point: an address or stride derived from this size would instead
        // have to be recomputed on every dispatch.
        //
        // size(), i.e. the LOGICAL size. Under a max-strided reservation the
        // allocation is deliberately larger, and it is the logical size that
        // varies per call and that the trip count must track.
        const at::Tensor& tensor = tensors[arg.tensor_id];
        TORCH_CHECK(arg.dim_index >= 0 && arg.dim_index < tensor.dim(),
                    "symbolic_args dim_index ", arg.dim_index,
                    " is out of range for the rank-", tensor.dim(),
                    " tensor at inputs_outputs[", arg.tensor_id, "]");
        const int64_t size = tensor.size(arg.dim_index);
        if (arg.granularity > 0 && arg.max_value > 0) {
          // The declaration's contract (#5179), the bundle's loop was built
          // for exactly it. The torch._check lines inside the trace and the
          // checks at the declaring transfer enforce it first; this is the
          // backstop for a path that reaches the launch without them. The
          // lower bound matters on its own: the backend clamps the trip count
          // only from above and its first slot always runs, so a size of 0
          // would pass a divisibility check alone and execute one tile.
          TORCH_CHECK(size >= arg.granularity && size <= arg.max_value &&
                          size % arg.granularity == 0,
                      "symbolic_args kDimension: size ", size,
                      " of inputs_outputs[", arg.tensor_id, "] along dim ",
                      arg.dim_index,
                      " is outside the declared contract, a positive multiple "
                      "of the granularity ",
                      arg.granularity, " up to the maximum ", arg.max_value);
        }
        resolved[i] = spyre_rt::HostComputeArg{
            spyre_rt::HostComputeArgKind::Value, spyre_rt::DeviceView{}, size};
        break;
      }
      default:
        TORCH_CHECK(false, "Unknown SymbolicArgKind value: ",
                    static_cast<int32_t>(arg.kind));
    }
  }
  return resolved;
}

void JobPlanStepHostCompute::construct(LaunchContext& ctx,
                                       const SpyreStream& stream) const {
  auto& args = ctx.host_compute_args;
  args.clear();

  // Cases 1 and 2 need no address args, input_buffer_ and ishape_ carry
  // the distinction into the host-compute call directly.
  if (input_buffer_ == nullptr && !(ishape_.size() == 1 && ishape_[0] == 0)) {
    if (!ctx.symbolic_args.empty()) {
      // Case 3a: typed symbolic args — delegate to resolveSymbolicArgs.
      args = resolveSymbolicArgs(ctx.inputs_outputs, ctx.symbolic_args);
    } else {
      // Case 3b: legacy: one Address arg per context tensor in order.
      for (const auto& tensor : ctx.inputs_outputs) {
        args.push_back(spyre_rt::HostComputeArg{
            spyre_rt::HostComputeArgKind::Address,
            *get_composite_address(tensor), /*value=*/0});
      }
    }
  }

  // Host compute runs inline on the caller (the vendor's
  // RuntimeStream::launchOperation(HostComputeParams*) resolves arguments
  // and calls deeptools::processComputeOnHostCommand[Fast] before
  // returning, Flex #1990): the bytes are ready before S_dev receives the
  // consuming compute, regardless of hazard tracking. The runtime owns the
  // produced Corrections and drops them at upload completion.
  rust::Box<spyre_rt::Corrections> corrections =
      GlobalRuntime::get()->host_compute(
          *handle_, spyre_rt::ByteSize{correction_size_},
          static_cast<const uint8_t*>(input_buffer_),
          rust::Slice<const spyre_rt::HostComputeArg>(args.data(),
                                                      args.size()));

  // Enforce that the produced correction size matches the expected device
  // allocation size.
  TORCH_DCHECK_EQ(spyre_rt::size(*corrections).raw, device_address_.size.raw);

#ifdef TORCH_SPYRE_FLEX
  // VF-only compound correction submission: PF's maxProgramCorrections() is 0
  // (the vendor base default, runtime_scheduler.hpp:141), so this branch is
  // unreachable on PF and the flex ProgramCorrection accumulation it needs is
  // not ported to spyre_rt. Fail loudly if a VF build ever reaches it rather
  // than silently dropping the batch.
  if (ctx.batch_program_corrections) {
    TORCH_CHECK(false,
                "batched program corrections (VF compound submission) are not "
                "ported to the spyre_rt runtime; PF never batches "
                "(maxProgramCorrections() == 0)");
  }
#endif

  stream.uploadCorrectionsStep(std::move(corrections), device_address_,
                               step_barrier(pipeline_barrier_));
}

void JobPlanStepHostCompute::write(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Correction size: " << correction_size_ << " bytes\n";
  os << "    Device address: "
     << std::string(spyre_rt::to_string(device_address_)) << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::ostream& operator<<(std::ostream& os, const JobPlan& plan) {
  os << "============ JobPlan =============\n";
  os << "Total steps: " << plan.steps.size() << "\n";

  // Job allocation
  if (!plan.job_allocation.empty()) {
    os << "Job allocation: "
       << std::string(spyre_rt::to_string(plan.job_allocation.front()->view()))
       << "\n";
  }
  for (size_t i = 0; i < plan.program_views.size(); ++i) {
    os << "Program " << i << ": "
       << std::string(spyre_rt::to_string(plan.program_views[i])) << "\n";
  }

  // Expected input shapes
  if (!plan.expected_input_shapes.empty()) {
    os << "Expected input shapes (" << plan.expected_input_shapes.size()
       << " tensors):\n";
    for (size_t i = 0; i < plan.expected_input_shapes.size(); ++i) {
      os << "  Input " << i << ": [";
      for (size_t j = 0; j < plan.expected_input_shapes[i].size(); ++j) {
        if (j > 0) os << ", ";
        os << plan.expected_input_shapes[i][j];
      }
      os << "]\n";
    }
  }

  // Pinned buffers
  os << "Pinned buffers: " << plan.pinned_buffers.size() << "\n";
  for (size_t i = 0; i < plan.pinned_buffers.size(); ++i) {
    const auto& buf = plan.pinned_buffers[i];
    os << "  Buffer " << i << ": ptr=" << buf.data() << ", size=" << buf.size()
       << " bytes\n";
  }

  // Detailed step information
  os << "\nDetailed Steps:\n";
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    os << "Step " << i << ": ";
    os << *plan.steps[i];
  }

  os << "==================================\n";
  return os;
}

StepKind classifyStep(const JobPlanStep& step) {
  if (dynamic_cast<const JobPlanStepHostCompute*>(&step)) {
    return StepKind::HostCompute;
  }
  if (dynamic_cast<const JobPlanStepH2D*>(&step)) {
    return StepKind::H2D;
  }
  if (dynamic_cast<const JobPlanStepD2H*>(&step)) {
    return StepKind::D2H;
  }
  if (dynamic_cast<const JobPlanStepCompute*>(&step)) {
    return StepKind::Compute;
  }
  return StepKind::Unknown;
}

const char* stepKindName(StepKind kind) {
  switch (kind) {
    case StepKind::HostCompute:
      return "HostCompute";
    case StepKind::H2D:
      return "H2D";
    case StepKind::D2H:
      return "D2H";
    case StepKind::Compute:
      return "Compute";
    case StepKind::Unknown:
    default:
      return "Unknown";
  }
}

StepKind stepKindFromName(const std::string& name) {
  if (name == "HostCompute") return StepKind::HostCompute;
  if (name == "H2D") return StepKind::H2D;
  if (name == "D2H") return StepKind::D2H;
  if (name == "Compute") return StepKind::Compute;
  if (name == "Unknown") return StepKind::Unknown;
  TORCH_CHECK(false, "Unknown StepKind name: ", name);
}

StreamRole streamRoleFromName(const std::string& name) {
  if (name == "Prep") return StreamRole::Prep;
  if (name == "Dev") return StreamRole::Dev;
  TORCH_CHECK(false, "Unknown StreamRole name: ", name, " (expected Prep/Dev)");
}

std::string checkJobPlanStepOrdering(const std::vector<StepKind>& kinds,
                                     const std::vector<StreamRole>& roles) {
  if (kinds.size() != roles.size()) {
    return "kinds/roles length mismatch";
  }

  // Gate: only validate plans built as HostCompute-led (the two-stream
  // correction triple). A plan without a HostCompute is legacy single-stream
  // and stays valid (backward-compat with the pre-overlap path: pure
  // ComputeOnDevice, standalone D2H, tensor .to() moves).
  bool has_host_compute = false;
  for (StepKind k : kinds) {
    if (k == StepKind::HostCompute) {
      has_host_compute = true;
    }
  }
  if (!has_host_compute) {
    return "";
  }

  // Project into the two per-stream subsequences, preserving plan order.
  std::vector<StepKind> prep;
  std::vector<StepKind> dev;
  for (size_t i = 0; i < kinds.size(); ++i) {
    if (roles[i] == StreamRole::Prep) {
      prep.push_back(kinds[i]);
    } else {
      dev.push_back(kinds[i]);
    }
  }

  auto name_at = [](const std::vector<StepKind>& seq, size_t i) {
    return std::string(i < seq.size() ? stepKindName(seq[i]) : "<end>");
  };

  // The contract is ordering-only, not an exact triple: prepare can emit longer
  // plans (e.g. HostCompute -> H2D -> Compute -> D2H), which project to
  // S_prep = [HostCompute, H2D] and S_dev = [Compute, D2H]. What must hold is
  // the leading-producer guarantee: prep produces (HostCompute -> H2D) before
  // dev consumes (Compute). On the HAZARD path torch-spyre emits no cross-
  // stream event steps; flex derives the RAW/WAR edges from these subsequences.

  // S_prep must BEGIN with HostCompute -> H2D and carry only {HostCompute, H2D}
  // (the persistent host-compute stream; see StreamRole in job_plan.h).
  {
    if (prep.size() < 2 || prep[0] != StepKind::HostCompute ||
        prep[1] != StepKind::H2D) {
      return "S_prep ordering violation: prep stream must begin with "
             "HostCompute -> H2D, got " +
             name_at(prep, 0) + " -> " + name_at(prep, 1);
    }
    for (size_t i = 2; i < prep.size(); ++i) {
      if (prep[i] != StepKind::HostCompute && prep[i] != StepKind::H2D) {
        return "S_prep ordering violation: " + name_at(prep, i) +
               " is not permitted on the prep stream (prep carries only "
               "HostCompute / H2D)";
      }
    }
  }

  // S_dev must BEGIN with Compute (leading-producer guarantee) and carry only
  // {Compute, D2H} (the device stream; see StreamRole in job_plan.h). No
  // HostCompute/H2D.
  {
    if (dev.empty() || dev[0] != StepKind::Compute) {
      return "S_dev ordering violation: device stream must begin with Compute, "
             "got " +
             name_at(dev, 0);
    }
    for (size_t i = 1; i < dev.size(); ++i) {
      if (dev[i] != StepKind::Compute && dev[i] != StepKind::D2H) {
        return "S_dev ordering violation: " + name_at(dev, i) +
               " is not permitted on the device stream (dev carries only "
               "Compute / D2H)";
      }
    }
  }

  return "";
}

}  // namespace spyre
