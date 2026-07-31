#include "pixie_compute/gpu_timer.hpp"

#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"

namespace pix
{

GpuTimer::GpuTimer(GpuContext &ctx, uint32_t pairCount)
    : m_device(ctx.device()), m_timestampPeriod(ctx.timestampPeriod()), m_pairCount(pairCount)
{
    if (pairCount == 0)
        throw GpuError("GpuTimer pairCount must be > 0");
    vk::QueryPoolCreateInfo poolInfo{};
    poolInfo.queryType = vk::QueryType::eTimestamp;
    poolInfo.queryCount = pairCount * 2;
    m_queryPool = vk::UniqueQueryPool(m_device.createQueryPoolUnique(poolInfo));
}

void GpuTimer::validateIndex(uint32_t pairIndex) const
{
    if (pairIndex >= m_pairCount)
        throw GpuError("GpuTimer pairIndex " + std::to_string(pairIndex) + " exceeds pairCount " +
                       std::to_string(m_pairCount));
}

void GpuTimer::resetAll(vk::CommandBuffer cmd)
{
    cmd.resetQueryPool(*m_queryPool, 0, m_pairCount * 2);
}

void GpuTimer::writeBegin(vk::CommandBuffer cmd, uint32_t pairIndex)
{
    validateIndex(pairIndex);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *m_queryPool, pairIndex * 2);
}

void GpuTimer::writeEnd(vk::CommandBuffer cmd, uint32_t pairIndex)
{
    validateIndex(pairIndex);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *m_queryPool,
                       pairIndex * 2 + 1);
}

double GpuTimer::readMs(uint32_t pairIndex) const
{
    validateIndex(pairIndex);
    uint64_t timestamps[2]{};
    auto result = m_device.getQueryPoolResults(
        *m_queryPool, pairIndex * 2, 2, sizeof(timestamps), timestamps, sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (result != vk::Result::eSuccess)
        throw GpuError("getQueryPoolResults failed (VkResult " +
                       std::to_string(static_cast<int>(result)) + ")");
    double ns = static_cast<double>(timestamps[1] - timestamps[0]) * m_timestampPeriod;
    return ns / 1.0e6;
}

void GpuTimer::begin(vk::CommandBuffer cmd)
{
    resetAll(cmd);
    writeBegin(cmd, 0);
}

void GpuTimer::end(vk::CommandBuffer cmd)
{
    writeEnd(cmd, 0);
}

} // namespace pix
