#include "pixie_compute/gpu_stream.hpp"

#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_timeline_semaphore.hpp"
#include "pixie_compute/utility.hpp"

namespace pix
{

GpuStream::GpuStream(GpuContext &ctx, GpuCommandBuffer::QueueType type)
    : m_ctx(&ctx), m_type(type), m_timeline(std::make_shared<GpuTimelineSemaphore>(ctx)),
      m_current(std::make_unique<GpuCommandBuffer>(ctx, type))
{
}

GpuStream::~GpuStream()
{
    try
    {
        wait();
    }
    catch (...)
    {
    }
}

void GpuStream::ensureRecording()
{
    if (!m_current)
        m_current = std::make_unique<GpuCommandBuffer>(*m_ctx, m_type);
    if (!m_current->recording())
        m_current->begin();
}

void GpuStream::begin()
{
    ensureRecording();
}

GpuEvent GpuStream::commit()
{
    ensureRecording();
    releaseCompletedSubmissions();
    const uint64_t value = m_timeline->next();
    GpuSubmitSync sync;
    for (const auto &event : m_pendingEvents)
    {
        if (!event.valid())
            throw GpuError("GpuStream cannot wait on an empty GpuEvent");
        sync.waits.push_back(
            GpuWaitSemaphore::makeTimeline(*event.m_state->semaphore, event.m_state->value));
    }
    sync.signals.push_back(GpuSignalSemaphore::makeTimeline(*m_timeline, value));
    m_current->submit(sync);
    m_inFlight.push_back({std::move(m_current), value});
    m_current = std::make_unique<GpuCommandBuffer>(*m_ctx, m_type);
    m_pendingEvents.clear();

    auto state = std::make_shared<GpuEvent::State>();
    state->semaphore = m_timeline;
    state->value = value;
    return GpuEvent(std::move(state));
}

void GpuStream::wait()
{
    if ((m_current && m_current->recording()) || !m_pendingEvents.empty())
        (void)commit();
    for (auto &inFlight : m_inFlight)
        inFlight.command->wait();
    m_inFlight.clear();
}

void GpuStream::releaseCompletedSubmissions()
{
    if (m_inFlight.empty())
        return;
    const uint64_t completed = m_timeline->gpuValue();
    auto it = m_inFlight.begin();
    while (it != m_inFlight.end())
    {
        if (it->value <= completed)
        {
            it->command->wait();
            it = m_inFlight.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void GpuStream::wait(const GpuEvent &event)
{
    if (!event.valid())
        throw GpuError("GpuStream cannot wait on an empty GpuEvent");
    m_pendingEvents.push_back(event);
}

void GpuStream::upload(GpuBuffer &buffer, const void *data, size_t size, size_t offset)
{
    ensureRecording();
    m_current->upload(buffer, data, size, offset);
}

void GpuStream::download(GpuBuffer &buffer, void *data, size_t size, size_t offset)
{
    ensureRecording();
    m_current->download(buffer, data, size, offset);
}

void GpuStream::copyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size,
                           vk::DeviceSize srcOffset, vk::DeviceSize dstOffset)
{
    ensureRecording();
    m_current->copyBuffer(src, dst, size, srcOffset, dstOffset);
}

void GpuStream::fillBuffer(vk::Buffer buffer, uint32_t value, vk::DeviceSize size,
                           vk::DeviceSize offset)
{
    ensureRecording();
    m_current->fillBuffer(buffer, value, size, offset);
}

void GpuStream::bufferBarrier(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
                              vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccess,
                              vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess)
{
    ensureRecording();
    m_current->bufferBarrier(buffer, offset, size, srcStage, srcAccess, dstStage, dstAccess);
}

void GpuStream::imageBarrier(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                             vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccess,
                             vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess,
                             vk::ImageAspectFlags aspectMask, uint32_t srcQueueFamily,
                             uint32_t dstQueueFamily)
{
    ensureRecording();
    m_current->imageBarrier(image, oldLayout, newLayout, srcStage, srcAccess, dstStage, dstAccess,
                            aspectMask, srcQueueFamily, dstQueueFamily);
}

void GpuStream::bind(const GpuComputePipeline &pipeline)
{
    ensureRecording();
    m_current->bind(pipeline);
}

void GpuStream::bind(const GpuComputePipeline &pipeline, uint32_t bindingSetIndex)
{
    ensureRecording();
    m_current->bind(pipeline, bindingSetIndex);
}

void GpuStream::pushConstants(const void *data, size_t size)
{
    ensureRecording();
    m_current->pushConstants(data, size);
}

void GpuStream::dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
{
    ensureRecording();
    m_current->dispatch(groupsX, groupsY, groupsZ);
}

void GpuStream::dispatch()
{
    ensureRecording();
    m_current->dispatch();
}

bool GpuStream::recording() const noexcept
{
    return m_current && m_current->recording();
}

GpuCommandBuffer &GpuStream::commandBuffer()
{
    ensureRecording();
    return *m_current;
}

} // namespace pix
