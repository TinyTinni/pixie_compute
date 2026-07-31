#include "pixie_compute/gpu_command_buffer.hpp"

#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_timeline_semaphore.hpp"
#include "pixie_compute/utility.hpp"

#include "error_utils.hpp"

#include <cstring>

namespace pix
{

GpuCommandBuffer::GpuCommandBuffer(GpuContext &ctx, QueueType type) : m_ctx(&ctx)
{
    auto device = ctx.device();
    vk::CommandPool pool;
    std::mutex *poolMutex;
    if (type == QueueType::Transfer)
    {
        pool = ctx.transferCommandPool();
        poolMutex = &ctx.transferQueueMutex();
    }
    else
    {
        pool = ctx.commandPool();
        poolMutex = &ctx.queueMutex();
    }
    {
        std::lock_guard lock(*poolMutex);
        vk::CommandBufferAllocateInfo allocInfo(pool, vk::CommandBufferLevel::ePrimary, 1);
        auto bufs = vkChecked([&] { return device.allocateCommandBuffersUnique(allocInfo); },
                              "vkAllocateCommandBuffers");
        m_cmd = std::move(bufs.front());
    }
    m_fence = vkChecked([&] { return device.createFenceUnique({}); }, "vkCreateFence");
    m_targetQueue = (type == QueueType::Transfer) ? ctx.transferQueue() : ctx.queue();
    m_queueMutex = poolMutex;
    // Wait on all stages so that transfer commands (copyBuffer, fillBuffer,
    // upload/download) recorded into a compute command buffer are also ordered
    // after any semaphore the submission waits on. A narrow wait stage (e.g.
    // eComputeShader) leaves transfer-stage work outside the semaphore's second
    // synchronization scope, producing read-after-write / write-after-write
    // hazards against the signaling submission.
    m_waitStage = (type == QueueType::Transfer) ? vk::PipelineStageFlagBits::eTransfer
                                                : vk::PipelineStageFlagBits::eAllCommands;
}

GpuCommandBuffer::~GpuCommandBuffer()
{
    // Wait for any in-flight submission before destroying resources
    if (m_submitted && m_fence)
    {
        try
        {
            if (m_ctx && m_ctx->device())
                (void)m_ctx->device().waitForFences(*m_fence, VK_TRUE, UINT64_MAX);
        }
        catch (...)
        {
            // Swallow exceptions in destructor
        }
        releaseStaging();
    }
}

GpuCommandBuffer::GpuCommandBuffer(GpuCommandBuffer &&other) noexcept
    : m_ctx(other.m_ctx), m_cmd(std::move(other.m_cmd)), m_fence(std::move(other.m_fence)),
      m_targetQueue(other.m_targetQueue), m_queueMutex(other.m_queueMutex),
      m_waitStage(other.m_waitStage), m_recording(other.m_recording), m_begun(other.m_begun),
      m_submitted(other.m_submitted),
      m_boundPipeline(other.m_boundPipeline), m_staging(std::move(other.m_staging)),
      m_pendingDownloads(std::move(other.m_pendingDownloads))
{
    other.m_ctx = nullptr;
    other.m_recording = false;
    other.m_begun = false;
    other.m_submitted = false;
    other.m_boundPipeline = nullptr;
}

GpuCommandBuffer &GpuCommandBuffer::operator=(GpuCommandBuffer &&other) noexcept
{
    if (this != &other)
    {
        // Wait for in-flight submission if any
        if (m_submitted && m_fence && m_ctx && m_ctx->device())
        {
            try
            {
                (void)m_ctx->device().waitForFences(*m_fence, VK_TRUE, UINT64_MAX);
            }
            catch (...)
            {
            }
            releaseStaging();
        }
        m_ctx = other.m_ctx;
        m_cmd = std::move(other.m_cmd);
        m_fence = std::move(other.m_fence);
        m_targetQueue = other.m_targetQueue;
        m_queueMutex = other.m_queueMutex;
        m_waitStage = other.m_waitStage;
        m_recording = other.m_recording;
        m_begun = other.m_begun;
        m_submitted = other.m_submitted;
        m_boundPipeline = other.m_boundPipeline;
        m_staging = std::move(other.m_staging);
        m_pendingDownloads = std::move(other.m_pendingDownloads);
        other.m_ctx = nullptr;
        other.m_recording = false;
        other.m_begun = false;
        other.m_submitted = false;
        other.m_boundPipeline = nullptr;
    }
    return *this;
}

void GpuCommandBuffer::begin()
{
    if (m_recording)
        throw GpuError("begin() called while already recording");
    vk::CommandBufferBeginInfo info;
    m_cmd->begin(info);
    m_recording = true;
    m_begun = true;
}

void GpuCommandBuffer::end()
{
    if (!m_recording)
        throw GpuError("end() called while not recording");
    m_cmd->end();
    m_recording = false;
}

void GpuCommandBuffer::reset()
{
    // Wait for any in-flight submission first
    if (m_submitted && m_fence && m_ctx && m_ctx->device())
    {
        auto result = m_ctx->device().waitForFences(*m_fence, VK_TRUE, UINT64_MAX);
        if (result != vk::Result::eSuccess)
            throw GpuError("waitForFences failed during reset");
        releaseStaging();
    }
    if (m_recording)
        end();
    m_cmd->reset();
    m_recording = false;
    m_submitted = false;
    m_begun = false;
    m_boundPipeline = nullptr;
}

void GpuCommandBuffer::submitAndWait(const GpuSubmitSync &sync)
{
    if (m_recording)
        end();
    if (!m_ctx)
        throw GpuError("submitAndWait on moved-from GpuCommandBuffer");
    if (!m_begun)
        throw GpuError("submitAndWait called before begin()");

    // Validate sync parameters
    if (sync.wait && sync.waitTimeline)
        throw GpuError("cannot specify both binary and timeline wait semaphores");
    if (sync.signal && sync.signalTimeline)
        throw GpuError("cannot specify both binary and timeline signal semaphores");

    auto device = m_ctx->device();
    device.resetFences(*m_fence);

    vk::PipelineStageFlags waitStage = m_waitStage;
    vk::Semaphore waitSem = sync.wait;
    vk::Semaphore signalSem = sync.signal;
    uint64_t waitValue = sync.waitValue;
    uint64_t signalValue = sync.signalValue;

    if (sync.waitTimeline)
    {
        waitSem = sync.waitTimeline->handle();
        waitValue = sync.waitValue;
    }
    if (sync.signalTimeline)
    {
        signalSem = sync.signalTimeline->handle();
        signalValue = sync.signalValue;
    }

    vk::SubmitInfo submitInfo(waitSem ? 1u : 0u, waitSem ? &waitSem : nullptr, &waitStage, 1,
                              &*m_cmd, signalSem ? 1u : 0u, signalSem ? &signalSem : nullptr);

    vk::TimelineSemaphoreSubmitInfo timelineInfo;
    if (sync.waitTimeline || sync.signalTimeline)
    {
        timelineInfo = vk::TimelineSemaphoreSubmitInfo(
            sync.waitTimeline ? 1u : 0u, sync.waitTimeline ? &waitValue : nullptr,
            sync.signalTimeline ? 1u : 0u, sync.signalTimeline ? &signalValue : nullptr);
        submitInfo.pNext = &timelineInfo;
    }

    {
        std::lock_guard lock(*m_queueMutex);
        if (m_targetQueue.submit(1, &submitInfo, *m_fence) != vk::Result::eSuccess)
            throw GpuError("vkQueueSubmit failed");
    }
    m_submitted = true;
    auto result = device.waitForFences(*m_fence, VK_TRUE, UINT64_MAX);
    if (result != vk::Result::eSuccess)
        throw GpuError("vkWaitForFences failed");

    finishDownloads();
}

void GpuCommandBuffer::submitAndWait(GpuTimelineSemaphore &waitTimeline, uint64_t waitValue)
{
    submitAndWait(GpuSubmitSync{.wait = {},
                                .signal = {},
                                .waitTimeline = &waitTimeline,
                                .waitValue = waitValue});
}

void GpuCommandBuffer::submit(const GpuSubmitSync &sync)
{
    if (m_recording)
        end();
    if (!m_ctx)
        throw GpuError("submit on moved-from GpuCommandBuffer");
    if (!m_begun)
        throw GpuError("submit called before begin()");
    if (m_submitted)
        throw GpuError("submit while previous submission in flight; call wait() or reset() first");

    // Validate sync parameters
    if (sync.wait && sync.waitTimeline)
        throw GpuError("cannot specify both binary and timeline wait semaphores");
    if (sync.signal && sync.signalTimeline)
        throw GpuError("cannot specify both binary and timeline signal semaphores");

    auto device = m_ctx->device();
    device.resetFences(*m_fence);

    vk::PipelineStageFlags waitStage = m_waitStage;
    vk::Semaphore waitSem = sync.wait;
    vk::Semaphore signalSem = sync.signal;
    uint64_t waitValue = sync.waitValue;
    uint64_t signalValue = sync.signalValue;

    if (sync.waitTimeline)
    {
        waitSem = sync.waitTimeline->handle();
        waitValue = sync.waitValue;
    }
    if (sync.signalTimeline)
    {
        signalSem = sync.signalTimeline->handle();
        signalValue = sync.signalValue;
    }

    vk::SubmitInfo submitInfo(waitSem ? 1u : 0u, waitSem ? &waitSem : nullptr, &waitStage, 1,
                              &*m_cmd, signalSem ? 1u : 0u, signalSem ? &signalSem : nullptr);

    vk::TimelineSemaphoreSubmitInfo timelineInfo;
    if (sync.waitTimeline || sync.signalTimeline)
    {
        timelineInfo = vk::TimelineSemaphoreSubmitInfo(
            sync.waitTimeline ? 1u : 0u, sync.waitTimeline ? &waitValue : nullptr,
            sync.signalTimeline ? 1u : 0u, sync.signalTimeline ? &signalValue : nullptr);
        submitInfo.pNext = &timelineInfo;
    }

    {
        std::lock_guard lock(*m_queueMutex);
        if (m_targetQueue.submit(1, &submitInfo, *m_fence) != vk::Result::eSuccess)
            throw GpuError("vkQueueSubmit failed");
    }
    m_submitted = true;
}

void GpuCommandBuffer::submit(vk::Semaphore waitSemaphore, uint64_t waitValue,
                              GpuTimelineSemaphore &signalTimeline, uint64_t signalValue)
{
    submit(GpuSubmitSync{.wait = waitSemaphore,
                         .signal = {},
                         .waitTimeline = nullptr,
                         .waitValue = waitValue,
                         .signalTimeline = &signalTimeline,
                         .signalValue = signalValue});
}

void GpuCommandBuffer::wait()
{
    if (!m_submitted)
        return;
    auto device = m_ctx->device();
    auto result = device.waitForFences(*m_fence, VK_TRUE, UINT64_MAX);
    if (result != vk::Result::eSuccess)
        throw GpuError("vkWaitForFences failed");

    finishDownloads();
}

// ---------------------------------------------------------------------------
// Buffer upload / download recording
// ---------------------------------------------------------------------------

void GpuCommandBuffer::upload(GpuBuffer &buf, const void *data, size_t size, size_t offset)
{
    if (!m_recording)
        throw GpuError("upload called outside begin()/end()");
    if (buf.m_type == GpuBuffer::Type::HostCoherent)
    {
        if (!buf.m_mapped)
            throw GpuError("upload on unmapped buffer");
        if (offset > buf.m_size || size > buf.m_size - offset)
            throw GpuError("upload exceeds buffer size");
        std::memcpy(static_cast<char *>(buf.m_mapped) + offset, data, size);
        return;
    }

    if (offset > buf.m_size || size > buf.m_size - offset)
        throw GpuError("upload exceeds buffer size");

    // Acquire staging and hold it until execution completes
    auto staging = m_ctx->acquireStagingBuffer(size);
    std::memcpy(staging.mapped, data, size);
    m_staging.push_back(staging);

    vk::BufferCopy region(0, offset, size);
    m_cmd->copyBuffer(staging.buffer, buf.m_buffer, 1, &region);
}

void GpuCommandBuffer::download(GpuBuffer &buf, void *data, size_t size, size_t offset)
{
    if (!m_recording)
        throw GpuError("download called outside begin()/end()");
    if (buf.m_type == GpuBuffer::Type::HostCoherent)
    {
        if (!buf.m_mapped)
            throw GpuError("download on unmapped buffer");
        if (offset > buf.m_size || size > buf.m_size - offset)
            throw GpuError("download exceeds buffer size");
        std::memcpy(data, static_cast<const char *>(buf.m_mapped) + offset, size);
        return;
    }

    if (offset > buf.m_size || size > buf.m_size - offset)
        throw GpuError("download exceeds buffer size");

    // Acquire staging and hold it until execution completes
    auto staging = m_ctx->acquireStagingBuffer(size);
    size_t stagingIndex = m_staging.size();
    m_staging.push_back(staging);

    vk::BufferCopy region(offset, 0, size);
    m_cmd->copyBuffer(buf.m_buffer, staging.buffer, 1, &region);

    m_pendingDownloads.push_back({stagingIndex, data, size});
}

void GpuCommandBuffer::finishDownloads()
{
    for (auto &dl : m_pendingDownloads)
    {
        if (dl.stagingIndex < m_staging.size())
        {
            std::memcpy(dl.hostDst, m_staging[dl.stagingIndex].mapped, dl.size);
        }
    }
    m_pendingDownloads.clear();
    releaseStaging();
}

void GpuCommandBuffer::releaseStaging()
{
    for (auto &st : m_staging)
    {
        if (st.buffer && m_ctx)
            m_ctx->releaseStagingBuffer(st);
    }
    m_staging.clear();
}

// ---------------------------------------------------------------------------
// Low-level commands
// ---------------------------------------------------------------------------

void GpuCommandBuffer::bufferBarrier(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
                                     vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccess,
                                     vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess)
{
    if (!m_recording)
        throw GpuError("bufferBarrier called outside begin()/end()");
    vk::BufferMemoryBarrier barrier(srcAccess, dstAccess, VK_QUEUE_FAMILY_IGNORED,
                                    VK_QUEUE_FAMILY_IGNORED, buffer, offset, size);
    m_cmd->pipelineBarrier(srcStage, dstStage, {}, {}, barrier, {});
}

void GpuCommandBuffer::imageBarrier(vk::Image image, vk::ImageLayout oldLayout,
                                    vk::ImageLayout newLayout, vk::PipelineStageFlags srcStage,
                                    vk::AccessFlags srcAccess, vk::PipelineStageFlags dstStage,
                                    vk::AccessFlags dstAccess, vk::ImageAspectFlags aspectMask,
                                    uint32_t srcQueueFamily, uint32_t dstQueueFamily)
{
    if (!m_recording)
        throw GpuError("imageBarrier called outside begin()/end()");
    vk::ImageSubresourceRange range(aspectMask, 0, VK_REMAINING_MIP_LEVELS, 0,
                                    VK_REMAINING_ARRAY_LAYERS);
    vk::ImageMemoryBarrier barrier(srcAccess, dstAccess, oldLayout, newLayout, srcQueueFamily,
                                   dstQueueFamily, image, range);
    m_cmd->pipelineBarrier(srcStage, dstStage, {}, {}, {}, {barrier});
}

void GpuCommandBuffer::copyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size,
                                  vk::DeviceSize srcOffset, vk::DeviceSize dstOffset)
{
    if (!m_recording)
        throw GpuError("copyBuffer called outside begin()/end()");
    vk::BufferCopy region(srcOffset, dstOffset, size);
    m_cmd->copyBuffer(src, dst, 1, &region);
}

void GpuCommandBuffer::fillBuffer(vk::Buffer buffer, uint32_t value, vk::DeviceSize size,
                                  vk::DeviceSize offset)
{
    if (!m_recording)
        throw GpuError("fillBuffer called outside begin()/end()");
    m_cmd->fillBuffer(buffer, offset, size, value);
}

void GpuCommandBuffer::bind(const GpuComputePipeline &pipeline)
{
    if (!m_recording)
        throw GpuError("bind called outside begin()/end()");
    m_cmd->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.handle());
    if (pipeline.descriptorSetCount() > 0)
    {
        // Access descriptor sets via friend
        m_cmd->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.layout(), 0,
                                  *pipeline.m_descriptorSets[0], {});
    }
    m_boundPipeline = &pipeline;
}

void GpuCommandBuffer::bind(const GpuComputePipeline &pipeline, uint32_t setIndex)
{
    if (!m_recording)
        throw GpuError("bind called outside begin()/end()");
    if (setIndex >= pipeline.descriptorSetCount())
        throw GpuError("bind: setIndex " + std::to_string(setIndex) +
                       " exceeds descriptor set count " +
                       std::to_string(pipeline.descriptorSetCount()));
    m_cmd->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.handle());
    m_cmd->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.layout(), 0,
                              *pipeline.m_descriptorSets[setIndex], {});
    m_boundPipeline = &pipeline;
}

void GpuCommandBuffer::pushConstants(const void *data, size_t size)
{
    if (!m_recording)
        throw GpuError("pushConstants called outside begin()/end()");
    if (!m_boundPipeline)
        throw GpuError("pushConstants called without a bound pipeline");
    if (size > m_boundPipeline->pushConstantSize())
        throw GpuError("push constant size exceeds pipeline layout range");
    m_cmd->pushConstants(m_boundPipeline->layout(), vk::ShaderStageFlagBits::eCompute, 0,
                         static_cast<uint32_t>(size), data);
}

void GpuCommandBuffer::pushConstants(const GpuComputePipeline &pipeline, const void *data,
                                     size_t size)
{
    if (!m_recording)
        throw GpuError("pushConstants called outside begin()/end()");
    pipeline.pushConstants(*m_cmd, data, size);
    m_boundPipeline = &pipeline;
}

void GpuCommandBuffer::dispatch()
{
    if (!m_recording)
        throw GpuError("dispatch called outside begin()/end()");
    if (!m_boundPipeline)
        throw GpuError("dispatch called without a bound pipeline");
    auto groups = m_boundPipeline->defaultGroups();
    m_cmd->dispatch(groups[0], groups[1], groups[2]);
}

void GpuCommandBuffer::dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
{
    if (!m_recording)
        throw GpuError("dispatch called outside begin()/end()");
    if (!m_boundPipeline)
        throw GpuError("dispatch called without a bound pipeline");
    m_cmd->dispatch(groupsX, groupsY, groupsZ);
}

void GpuCommandBuffer::dispatch(const GpuComputePipeline &pipeline)
{
    if (!m_recording)
        throw GpuError("dispatch called outside begin()/end()");
    pipeline.dispatch(*m_cmd);
    m_boundPipeline = &pipeline;
}

void GpuCommandBuffer::dispatch(const GpuComputePipeline &pipeline, uint32_t groupsX,
                                uint32_t groupsY, uint32_t groupsZ)
{
    if (!m_recording)
        throw GpuError("dispatch called outside begin()/end()");
    pipeline.dispatch(*m_cmd, groupsX, groupsY, groupsZ);
    m_boundPipeline = &pipeline;
}

// ---------------------------------------------------------------------------
// Debug-utils object names and command labels
// ---------------------------------------------------------------------------

void GpuCommandBuffer::setDebugName(const std::string &name)
{
    m_ctx->setDebugName(*m_cmd, name);
}

void GpuCommandBuffer::beginLabel(const std::string &name, const std::array<float, 4> &color)
{
    if (!m_recording)
        throw GpuError("beginLabel() called while not recording");
    if (!m_ctx->debugUtilsEnabled())
        return;
    vk::DebugUtilsLabelEXT label;
    label.setPLabelName(name.c_str());
    label.setColor(color);
    m_cmd->beginDebugUtilsLabelEXT(label);
}

void GpuCommandBuffer::endLabel()
{
    if (!m_recording)
        throw GpuError("endLabel() called while not recording");
    if (!m_ctx->debugUtilsEnabled())
        return;
    m_cmd->endDebugUtilsLabelEXT();
}

GpuCommandBuffer::ScopedLabel::ScopedLabel(GpuCommandBuffer &cmd, const std::string &name,
                                           const std::array<float, 4> &color)
    : m_cmd(&cmd)
{
    cmd.beginLabel(name, color);
}

GpuCommandBuffer::ScopedLabel::~ScopedLabel()
{
    // noexcept: skip if not recording (e.g., after submit auto-end)
    if (m_cmd && m_cmd->recording())
    {
        try
        {
            m_cmd->endLabel();
        }
        catch (...)
        {
            // Swallow exceptions in destructor
        }
    }
}

GpuCommandBuffer::ScopedLabel::ScopedLabel(ScopedLabel &&other) noexcept : m_cmd(other.m_cmd)
{
    other.m_cmd = nullptr;
}

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline, uint32_t groupsX,
                     uint32_t groupsY, uint32_t groupsZ)
{
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);
    cmd.dispatch(groupsX, groupsY, groupsZ);
    cmd.submitAndWait();
}

void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline, const void *pushConstantData,
                     size_t pushConstantSize, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
{
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);
    cmd.pushConstants(pushConstantData, pushConstantSize);
    cmd.dispatch(groupsX, groupsY, groupsZ);
    cmd.submitAndWait();
}

void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline)
{
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);
    cmd.dispatch();
    cmd.submitAndWait();
}

} // namespace pix
