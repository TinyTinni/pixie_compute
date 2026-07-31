#pragma once

#include "pixie_compute/detail/vulkan_include.hpp"

#include <cstdint>

namespace pix
{

class GpuContext;

/// GPU timestamp query helper using Vulkan timestamp queries.
class GpuTimer
{
    public:
    /// Construct a timer with the given number of timestamp pairs (begin/end per pair).
    explicit GpuTimer(GpuContext &ctx, uint32_t pairCount = 1);

    GpuTimer(const GpuTimer &) = delete;
    GpuTimer &operator=(const GpuTimer &) = delete;
    GpuTimer(GpuTimer &&) = default;
    GpuTimer &operator=(GpuTimer &&) = default;

    /// Reset all query pool entries.
    void resetAll(vk::CommandBuffer cmd);

    /// Write a begin timestamp for the given pair index.
    void writeBegin(vk::CommandBuffer cmd, uint32_t pairIndex);

    /// Write an end timestamp for the given pair index.
    void writeEnd(vk::CommandBuffer cmd, uint32_t pairIndex);

    /// Read the elapsed time in milliseconds for the given pair index. Blocks until results are
    /// available.
    double readMs(uint32_t pairIndex) const;

    /// Convenience: reset all queries + writeBegin(0).
    void begin(vk::CommandBuffer cmd);
    /// Convenience: writeEnd(0).
    void end(vk::CommandBuffer cmd);

    uint32_t pairCount() const noexcept { return m_pairCount; }

    private:
    void validateIndex(uint32_t pairIndex) const;

    vk::Device m_device;
    float m_timestampPeriod;
    uint32_t m_pairCount;
    vk::UniqueQueryPool m_queryPool;
};

} // namespace pix
