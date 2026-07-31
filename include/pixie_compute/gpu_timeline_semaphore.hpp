#pragma once

#include <cstdint>
#include <string>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>
#include <vulkan/vulkan.hpp>

namespace pix
{

class GpuContext;

/// RAII timeline semaphore with auto-incrementing counter.
/// Timeline semaphores use a monotonically increasing 64-bit value instead of
/// the binary signaled/unsignaled states. Submit signals up to a value; other
/// submits wait until the counter reaches >= that value. No reset needed.
///
/// Usage:
///   GpuTimelineSemaphore sem(ctx);       // starts at 0
///   uint64_t val = sem.next();           // returns 1 (for signalValue)
///   uploadCmd.submit({.signalTimeline = &sem, .signalValue = val});
///   computeCmd.submitAndWait({.waitTimeline = &sem, .waitValue = val});
struct GpuTimelineSemaphore
{
    /// Create a timeline semaphore starting at initialValue.
    explicit GpuTimelineSemaphore(GpuContext &ctx, uint64_t initialValue = 0);
    ~GpuTimelineSemaphore();

    GpuTimelineSemaphore(const GpuTimelineSemaphore &) = delete;
    GpuTimelineSemaphore &operator=(const GpuTimelineSemaphore &) = delete;
    GpuTimelineSemaphore(GpuTimelineSemaphore &&other) noexcept;
    GpuTimelineSemaphore &operator=(GpuTimelineSemaphore &&other) noexcept;

    vk::Semaphore handle() const noexcept { return *m_semaphore; }

    /// The value that will be used on the next signal (calling next() returns this + 1).
    uint64_t currentValue() const noexcept { return m_value; }

    /// Increment the counter and return the new value. Use as signalValue.
    uint64_t next() { return ++m_value; }

    /// Query the current GPU counter value. Blocks until the query completes.
    uint64_t gpuValue() const;

    /// Block on the host until the semaphore reaches the given value.
    /// @param timeoutNs Timeout in nanoseconds (default: UINT64_MAX = wait forever).
    /// @return true if the value was reached, false on timeout.
    bool hostWait(uint64_t value, uint64_t timeoutNs = UINT64_MAX) const;

    /// Assign a debug name to the semaphore (VK_EXT_debug_utils). No-op when the
    /// extension is unavailable.
    void setDebugName(const std::string &name);

    private:
    vk::UniqueSemaphore m_semaphore;
    uint64_t m_value = 0;
    GpuContext *m_ctx = nullptr;
};

} // namespace pix
