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

#include <ATen/ATen.h>
#include <c10/core/Stream.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "job_plan.h"
#include "module.h"
#include "spyre_error.h"
#include "spyre_kernel.h"

namespace spyre {

class SpyreStream {
 private:
  c10::Stream stream_;

 public:
  explicit SpyreStream(c10::Stream stream);
  SpyreStream();

  // Stream properties
  c10::StreamId id() const;
  c10::Device device() const;
  int priority() const;

  // Synchronization
  bool query() const;        // Check if work completed
  void synchronize() const;  // Block until work done

  void copyAsync(const at::Tensor& src, const at::Tensor& dst) const;
  void copyProgramAsync(void* prog_cpu_ptr,
                        const spyre_rt::DeviceView* device_address) const;

  void launch(const JobPlan& plan, const std::vector<at::Tensor>& args,
              std::vector<SymbolicArg> symbolic_args = {}) const;

  // The typed spyre_rt entry points JobPlan steps call at launch time. These
  // are the single chokepoint through which torch-spyre submits work to the
  // runtime; the raw spyre_rt::Stream handle never escapes SpyreStream.
  // The barrier argument is the caller step's pipeline_barrier_ flag.
  void copyH2D(const uint8_t* src, spyre_rt::ByteSize src_capacity,
               spyre_rt::DeviceView dst, spyre_rt::PipelineBarrier barrier) const;
  void copyD2H(spyre_rt::DeviceView src, uint8_t* dst,
               spyre_rt::ByteSize dst_capacity,
               spyre_rt::PipelineBarrier barrier) const;
  void launchComputeStep(
      spyre_rt::DeviceView program, spyre_rt::BootstrapOffset bootstrap,
      rust::Slice<const spyre_rt::DeviceView> tensors, rust::Str name,
      spyre_rt::PipelineBarrier barrier) const;
  void uploadCorrectionsStep(spyre_rt::CorrectionsHandle corrections,
                             spyre_rt::DeviceView dst,
                             spyre_rt::PipelineBarrier barrier) const;
  spyre_rt::CorrectionsHandle hostComputeStep(
      const spyre_rt::HostComputeHandle& handle,
      spyre_rt::ByteSize correction_size, const uint8_t* input,
      rust::Slice<const spyre_rt::HostComputeArg> args) const;
  // Device-side fill (the vendor's FillAsync MEMORY_FILL DMA).
  void fill(spyre_rt::DeviceView dst, double value,
            spyre_rt::DataFormat format) const;

  // Tensor parallelism (spyre_rt): send `src` card to card to rank `to` at
  // `parity`, and hold this stream until every peer's message at `parity`
  // has landed (distributed_rt/spyre_rt_ccl.cpp).
  void tpSend(const spyre_rt::Runtime& rt, uint8_t to, uint8_t parity,
              spyre_rt::DeviceView src) const;
  void tpWait(const spyre_rt::Runtime& rt, uint8_t parity) const;

  // Conversions
  c10::Stream unwrap() const;

  // Returns the error state of this stream without exposing the underlying
  // runtime handle to callers.
  SpyreStreamError getError() const;

 private:
  const spyre_rt::Stream* resolveRuntimeHandle() const;
  void copyAsyncImpl(void* cpu_ptr, size_t host_capacity,
                     const spyre_rt::DeviceView* device_address,
                     const DataConversionInfo* dci, bool host2device,
                     std::unique_ptr<spyre_rt::HostKeepAlive> keep = nullptr) const;
};

/**
 * Get a stream from the global stream pool.
 * Streams are allocated round-robin from a pool maintained by torch-spyre.
 *
 * @param device Device to allocate stream on (default: current device)
 * @param priority Stream priority: 0 (normal) or -1 (high)
 * @return A SpyreStream from the pool
 */
SpyreStream getStreamFromPool(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1),
    int priority = 0);

/**
 * Get the default stream for a device.
 * The default stream is stream ID 0 and is always available.
 *
 * @param device Device to get default stream for
 * @return The default SpyreStream (stream ID 0)
 */
SpyreStream getDefaultStream(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1));

/**
 * Get a host compute stream for a device (round-robin).
 * Host compute streams are IDs kHostComputeStreamStartPerDevice and up; the
 * count is set by TORCH_SPYRE_NUM_HOST_COMPUTE_STREAMS.
 *
 * @param device Device to get a host compute stream for
 * @return The next host compute SpyreStream in round-robin order
 */
SpyreStream getHostComputeStream(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1));

/**
 * Get a specific host compute stream by stream ID.
 *
 * @param id Stream ID of the host compute stream
 * @param device Device the stream belongs to
 * @return The host compute SpyreStream with the given ID
 */
SpyreStream getHostComputeStreamById(
    c10::StreamId id,
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1));

/**
 * Get the runtime-level default stream for a device.
 * The default stream is stream ID 0 and is always available.
 *
 * @param device Device to get default stream for
 * @return The default runtime Stream (stream ID 0)
 */
const spyre_rt::Stream* getDefaultStreamRuntimeHandle(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1));

/**
 * Get the current stream for a device (thread-local).
 * Each thread maintains its own current stream per device.
 *
 * @param device Device to get current stream for
 * @return The current SpyreStream for this thread
 */
SpyreStream getCurrentStream(
    c10::Device device = c10::Device(c10::DeviceType::PrivateUse1, -1));

/**
 * Set the current stream for a device (thread-local).
 *
 * @param stream Stream to set as current
 * @return The previous current stream
 */
SpyreStream setCurrentStream(SpyreStream stream);

void synchronizeDevice(c10::optional<c10::Device> device);

}  // namespace spyre
