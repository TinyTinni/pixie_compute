#include "pixie_compute/gpu_timeline_semaphore.hpp"

#include "pixie_compute/gpu_context.hpp"

#include "pixie_compute/utility.hpp"

namespace pix
{

GpuTimelineSemaphore::GpuTimelineSemaphore(GpuContext &ctx, uint64_t initialValue)
    : m_value(initialValue), m_ctx(&ctx)
{
    vk::SemaphoreTypeCreateInfo timelineInfo(vk::SemaphoreType::eTimeline, initialValue);
    vk::SemaphoreCreateInfo semInfo({}, &timelineInfo);
    m_semaphore = ctx.device().createSemaphoreUnique(semInfo);
}

GpuTimelineSemaphore::~GpuTimelineSemaphore() = default;

GpuTimelineSemaphore::GpuTimelineSemaphore(GpuTimelineSemaphore &&other) noexcept
    : m_semaphore(std::move(other.m_semaphore)), m_value(other.m_value), m_ctx(other.m_ctx)
{
    other.m_ctx = nullptr;
}

GpuTimelineSemaphore &GpuTimelineSemaphore::operator=(GpuTimelineSemaphore &&other) noexcept
{
    if (this != &other)
    {
        m_semaphore = std::move(other.m_semaphore);
        m_value = other.m_value;
        m_ctx = other.m_ctx;
        other.m_ctx = nullptr;
    }
    return *this;
}

void GpuTimelineSemaphore::setDebugName(const std::string &name)
{
    if (!m_ctx)
        return;
    m_ctx->setDebugName(*m_semaphore, name);
}

uint64_t GpuTimelineSemaphore::gpuValue() const
{
    if (!m_ctx)
        throw GpuError("gpuValue on moved-from GpuTimelineSemaphore");
    return m_ctx->device().getSemaphoreCounterValue(*m_semaphore);
}

bool GpuTimelineSemaphore::hostWait(uint64_t value, uint64_t timeoutNs) const
{
    if (!m_ctx)
        throw GpuError("hostWait on moved-from GpuTimelineSemaphore");
    vk::SemaphoreWaitInfo info({}, *m_semaphore, value);
    auto result = m_ctx->device().waitSemaphores(info, timeoutNs);
    if (result == vk::Result::eSuccess)
        return true;
    if (result == vk::Result::eTimeout)
        return false;
    throw GpuError("vkWaitSemaphores failed (VkResult " +
                   std::to_string(static_cast<int>(result)) + ")");
}

} // namespace pix
