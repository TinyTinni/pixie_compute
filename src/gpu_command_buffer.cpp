#include "pixie_compute/gpu_command_buffer.hpp"

#include "error_utils.hpp"
#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_timeline_semaphore.hpp"
#include "pixie_compute/utility.hpp"

#include <cstring>

namespace pix
{

struct GpuDownload::State
{
    std::shared_ptr<GpuCommandBuffer> command;
    std::vector<std::byte> bytes;
    bool waited = false;
};

GpuDownload::~GpuDownload()
{
    if (m_state && m_state->command)
    {
        try
        {
            m_state->command->wait();
        }
        catch (...)
        {
        }
    }
}

GpuDownload::GpuDownload(GpuDownload &&other) noexcept : m_state(std::move(other.m_state))
{
}

GpuDownload &GpuDownload::operator=(GpuDownload &&other) noexcept
{
    if (this != &other)
    {
        m_state = std::move(other.m_state);
    }
    return *this;
}

void GpuDownload::wait()
{
    if (!m_state || m_state->waited)
        return;
    if (m_state->command)
        m_state->command->wait();
    m_state->waited = true;
}

const std::vector<std::byte> &GpuDownload::bytes()
{
    wait();
    if (!m_state)
        throw GpuError("accessing an empty GpuDownload");
    return m_state->bytes;
}

std::vector<std::byte> GpuDownload::get()
{
    const auto &result = bytes();
    return result;
}

GpuDownload GpuBuffer::downloadAsync(size_t size, size_t offset)
{
    return downloadAsync(size, offset, GpuSubmitSync{});
}

GpuDownload GpuBuffer::downloadAsync(size_t size, size_t offset, const GpuSubmitSync &sync)
{
    if (offset > m_size || size > m_size - offset)
        throw GpuError("download exceeds buffer size");

    auto state = std::make_shared<GpuDownload::State>();
    state->bytes.resize(size);
    if (size == 0)
    {
        state->waited = true;
        return GpuDownload(std::move(state));
    }

    if (m_type == GpuBuffer::Type::HostCoherent)
    {
        if (!m_mapped)
            throw GpuError("download on unmapped buffer");
        std::memcpy(state->bytes.data(), static_cast<const char *>(m_mapped) + offset, size);
        state->waited = true;
        return GpuDownload(std::move(state));
    }

    state->command = std::make_shared<GpuCommandBuffer>(*m_ctx);
    state->command->begin();
    state->command->download(*this, state->bytes.data(), size, offset);
    state->command->end();
    state->command->submit(sync);
    return GpuDownload(std::move(state));
}

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
            // Copy any pending download data to its host destination and return
            // the staging buffers to the pool before the command buffer dies.
            finishDownloads();
        }
        catch (...)
        {
            // Swallow exceptions in destructor
        }
    }
    // Return any staging acquired for a recording that never completed.
    releaseStaging();
}

GpuCommandBuffer::GpuCommandBuffer(GpuCommandBuffer &&other) noexcept
    : m_ctx(other.m_ctx), m_cmd(std::move(other.m_cmd)), m_fence(std::move(other.m_fence)),
      m_targetQueue(other.m_targetQueue), m_queueMutex(other.m_queueMutex),
      m_recording(other.m_recording), m_begun(other.m_begun), m_submitted(other.m_submitted),
      m_boundPipeline(other.m_boundPipeline), m_staging(std::move(other.m_staging)),
      m_pendingDownloads(std::move(other.m_pendingDownloads)),
      m_recordedLabels(std::move(other.m_recordedLabels)),
      m_labelStack(std::move(other.m_labelStack))
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
        // Swap states: the moved-from object now holds any in-flight submission and
        // staging we owned, and its destructor waits for completion and releases it.
        using std::swap;
        swap(m_ctx, other.m_ctx);
        swap(m_cmd, other.m_cmd);
        swap(m_fence, other.m_fence);
        swap(m_targetQueue, other.m_targetQueue);
        swap(m_queueMutex, other.m_queueMutex);
        swap(m_recording, other.m_recording);
        swap(m_begun, other.m_begun);
        swap(m_submitted, other.m_submitted);
        swap(m_boundPipeline, other.m_boundPipeline);
        swap(m_staging, other.m_staging);
        swap(m_pendingDownloads, other.m_pendingDownloads);
        swap(m_recordedLabels, other.m_recordedLabels);
        swap(m_labelStack, other.m_labelStack);
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
        // Copy any pending download data to its host destination before the
        // staging buffers are returned to the pool.
        finishDownloads();
    }
    if (m_recording)
        end();
    m_cmd->reset();
    m_recording = false;
    m_submitted = false;
    m_begun = false;
    m_boundPipeline = nullptr;
    m_recordedLabels.clear();
    m_labelStack.clear();
    // Release staging left over from a recording that never submitted.
    releaseStaging();
}

void GpuCommandBuffer::submitAndWait(const GpuSubmitSync &sync)
{
    submit(sync);
    wait();
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

    auto device = m_ctx->device();
    device.resetFences(*m_fence);

    std::vector<vk::Semaphore> waitSems;
    std::vector<vk::PipelineStageFlags> waitStages;
    std::vector<uint64_t> waitValues;
    std::vector<vk::Semaphore> signalSems;
    std::vector<uint64_t> signalValues;
    waitSems.reserve(sync.waits.size());
    waitStages.reserve(sync.waits.size());
    signalSems.reserve(sync.signals.size());

    uint32_t timelineWaitCount = 0;
    uint32_t timelineSignalCount = 0;
    for (const auto &wait : sync.waits)
    {
        if (wait.timeline)
        {
            waitSems.push_back(wait.timeline->handle());
            waitStages.push_back(wait.stage);
            waitValues.push_back(wait.value);
            ++timelineWaitCount;
        }
        else if (wait.binary)
        {
            waitSems.push_back(wait.binary);
            waitStages.push_back(wait.stage);
        }
    }
    for (const auto &signal : sync.signals)
    {
        if (signal.timeline)
        {
            signalSems.push_back(signal.timeline->handle());
            signalValues.push_back(signal.value);
            ++timelineSignalCount;
        }
        else if (signal.binary)
        {
            signalSems.push_back(signal.binary);
        }
    }

    vk::SubmitInfo submitInfo(static_cast<uint32_t>(waitSems.size()), waitSems.data(),
                              waitStages.data(), 1, &*m_cmd,
                              static_cast<uint32_t>(signalSems.size()), signalSems.data());

    vk::TimelineSemaphoreSubmitInfo timelineInfo;
    if (timelineWaitCount > 0 || timelineSignalCount > 0)
    {
        timelineInfo = vk::TimelineSemaphoreSubmitInfo(timelineWaitCount, waitValues.data(),
                                                       timelineSignalCount, signalValues.data());
        submitInfo.setPNext(&timelineInfo);
    }

    {
        std::lock_guard lock(*m_queueMutex);
        const auto result = m_targetQueue.submit(1, &submitInfo, *m_fence);
        if (result != vk::Result::eSuccess)
        {
            m_ctx->reportGpuFailure(result, "vkQueueSubmit", m_recordedLabels);
            throw GpuError("vkQueueSubmit failed");
        }
    }
    m_submitted = true;
}

void GpuCommandBuffer::wait()
{
    if (!m_submitted)
        return;
    auto device = m_ctx->device();
    auto result = device.waitForFences(*m_fence, VK_TRUE, UINT64_MAX);
    if (result != vk::Result::eSuccess)
    {
        m_ctx->reportGpuFailure(result, "vkWaitForFences", m_recordedLabels);
        throw GpuError("vkWaitForFences failed");
    }
    m_submitted = false;

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
    if (pipeline.bindingSetCount() > 0)
    {
        // Access descriptor sets via friend
        m_cmd->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.layout(), 0,
                                  *pipeline.m_descriptorSets[0], {});
    }
    m_boundPipeline = &pipeline;
}

void GpuCommandBuffer::bind(const GpuComputePipeline &pipeline, uint32_t bindingSetIndex)
{
    if (!m_recording)
        throw GpuError("bind called outside begin()/end()");
    if (bindingSetIndex >= pipeline.bindingSetCount())
        throw GpuError("bind: bindingSetIndex " + std::to_string(bindingSetIndex) +
                       " exceeds binding set count " + std::to_string(pipeline.bindingSetCount()));
    m_cmd->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.handle());
    m_cmd->bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.layout(), 0,
                              *pipeline.m_descriptorSets[bindingSetIndex], {});
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
    m_recordedLabels.push_back(name);
    m_labelStack.push_back(name);
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
    if (!m_labelStack.empty())
        m_labelStack.pop_back();
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

void detail::oneShotDispatchWithDefaultGroups(GpuContext &ctx, GpuComputePipeline &pipeline,
                                              const void *pushConstantData, size_t pushConstantSize)
{
    const auto groups = pipeline.defaultGroups();
    oneShotDispatch(ctx, pipeline, pushConstantData, pushConstantSize, groups[0], groups[1],
                    groups[2]);
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
