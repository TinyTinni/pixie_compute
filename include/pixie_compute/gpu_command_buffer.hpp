#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <span>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>
#include <vulkan/vulkan.hpp>

namespace pix
{

class GpuContext;
class GpuComputePipeline;
class GpuBuffer;
class GpuTimelineSemaphore;
struct StagingBuffer;

/// Synchronization parameters for command buffer submission.
struct GpuSubmitSync
{
    /// Binary semaphore to wait on before execution begins.
    vk::Semaphore wait = {};
    /// Binary semaphore to signal after execution completes.
    vk::Semaphore signal = {};
    /// Timeline semaphore to wait on (mutually exclusive with binary wait).
    GpuTimelineSemaphore *waitTimeline = nullptr;
    /// Timeline value to wait for (when waitTimeline is set).
    uint64_t waitValue = 0;
    /// Timeline semaphore to signal (mutually exclusive with binary signal).
    GpuTimelineSemaphore *signalTimeline = nullptr;
    /// Timeline value to signal (when signalTimeline is set).
    uint64_t signalValue = 0;
};

/// Records and submits Vulkan command buffers.
class GpuCommandBuffer
{
    public:
    enum class QueueType
    {
        Compute,
        Transfer
    };

    explicit GpuCommandBuffer(GpuContext &ctx, QueueType type = QueueType::Compute);
    ~GpuCommandBuffer();

    GpuCommandBuffer(const GpuCommandBuffer &) = delete;
    GpuCommandBuffer &operator=(const GpuCommandBuffer &) = delete;
    GpuCommandBuffer(GpuCommandBuffer &&) noexcept;
    GpuCommandBuffer &operator=(GpuCommandBuffer &&) noexcept;

    /// Begin recording commands. Throws GpuError if already recording.
    void begin();
    /// End recording commands. Throws GpuError if not recording.
    void end();
    /// Reset the command buffer. Waits for any in-flight submission first, then
    /// resets. Ends recording if needed.
    void reset();

    /// Submit and block until execution completes. Auto-finishes any pending downloads.
    /// Pass an empty GpuSubmitSync{} for no synchronization.
    void submitAndWait(const GpuSubmitSync &sync = {});
    [[deprecated("use GpuSubmitSync")]]
    void submitAndWait(GpuTimelineSemaphore &waitTimeline, uint64_t waitValue);

    /// Submit without blocking. Stores fence internally for wait().
    void submit(const GpuSubmitSync &sync = {});
    [[deprecated("use GpuSubmitSync")]]
    void submit(vk::Semaphore waitSemaphore, uint64_t waitValue,
                GpuTimelineSemaphore &signalTimeline, uint64_t signalValue);

    /// Block until the last submit() completes. Auto-finishes pending downloads.
    void wait();

    /// Record a buffer upload: copies CPU data to staging, records staging→buffer GPU copy.
    /// For HostCoherent buffers, does a direct memcpy (no GPU submission needed).
    void upload(GpuBuffer &buf, const void *data, size_t size, size_t offset = 0);

    /// Record a buffer download: records buffer→staging GPU copy, stores pending host copy.
    /// For HostCoherent buffers, does a direct memcpy.
    /// Call finishDownloads() or wait()/submitAndWait() after GPU completes.
    void download(GpuBuffer &buf, void *data, size_t size, size_t offset = 0);

    /// Copy pending download data from staging to host pointers.
    /// Called automatically by wait() and submitAndWait().
    void finishDownloads();

    template <typename T>
    void upload(GpuBuffer &buf, std::span<const T> data, size_t byteOffset = 0)
    {
        upload(buf, data.data(), data.size_bytes(), byteOffset);
    }

    template <typename T> void download(GpuBuffer &buf, std::span<T> data, size_t byteOffset = 0)
    {
        download(buf, data.data(), data.size_bytes(), byteOffset);
    }

    /// Insert a pipeline barrier for a buffer range.
    void bufferBarrier(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
                       vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccess,
                       vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess);

    /// Insert a pipeline barrier for an image subresource range.
    void imageBarrier(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                      vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccess,
                      vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess,
                      vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor,
                      uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED,
                      uint32_t dstQueueFamily = VK_QUEUE_FAMILY_IGNORED);

    /// Copy data between two buffers.
    void copyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size,
                    vk::DeviceSize srcOffset = 0, vk::DeviceSize dstOffset = 0);

    /// Fill a buffer range with a 32-bit pattern.
    void fillBuffer(vk::Buffer buffer, uint32_t value, vk::DeviceSize size,
                    vk::DeviceSize offset = 0);

    /// Bind a compute pipeline for subsequent dispatch calls.
    void bind(const GpuComputePipeline &pipeline);
    /// Bind a specific descriptor set from the bound pipeline.
    void bind(const GpuComputePipeline &pipeline, uint32_t setIndex);
    /// Push constants to the bound pipeline.
    void pushConstants(const void *data, size_t size);
    [[deprecated("use bind() then pushConstants(data, size)")]]
    void pushConstants(const GpuComputePipeline &pipeline, const void *data, size_t size);

    template <typename T> void pushConstants(const T &pc)
    {
        pushConstants(&pc, sizeof(T));
    }

    /// Dispatch using the bound pipeline's default group counts.
    void dispatch();
    /// Dispatch with explicit workgroup counts.
    void dispatch(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1);
    [[deprecated("use bind() then dispatch(groups)")]]
    void dispatch(const GpuComputePipeline &pipeline);
    [[deprecated("use bind() then dispatch(groups)")]]
    void dispatch(const GpuComputePipeline &pipeline, uint32_t groupsX, uint32_t groupsY,
                  uint32_t groupsZ);

    vk::CommandBuffer handle() const noexcept { return *m_cmd; }
    bool recording() const noexcept { return m_recording; }

    /// Assign a debug name to the command buffer (VK_EXT_debug_utils). No-op when
    /// the extension is unavailable.
    void setDebugName(const std::string &name);

    /// Insert a nested debug-utils label region. beginLabel()/endLabel() must be
    /// balanced within one recording. No-ops when debug-utils is unavailable.
    void beginLabel(const std::string &name,
                    const std::array<float, 4> &color = {1.0f, 1.0f, 1.0f, 1.0f});
    void endLabel();

    /// RAII guard that calls beginLabel() on construction and endLabel() on destruction.
    /// Move-only; the moved-from guard does nothing on destruction.
    class ScopedLabel
    {
        public:
        ScopedLabel(GpuCommandBuffer &cmd, const std::string &name,
                    const std::array<float, 4> &color = {1.0f, 1.0f, 1.0f, 1.0f});
        ~ScopedLabel();

        ScopedLabel(const ScopedLabel &) = delete;
        ScopedLabel &operator=(const ScopedLabel &) = delete;
        ScopedLabel(ScopedLabel &&other) noexcept;
        ScopedLabel &operator=(ScopedLabel &&) = delete;

        private:
        GpuCommandBuffer *m_cmd;
    };

    private:
    struct PendingDownload
    {
        size_t stagingIndex;
        void *hostDst;
        size_t size;
    };

    void releaseStaging();

    GpuContext *m_ctx;
    vk::UniqueCommandBuffer m_cmd;
    vk::UniqueFence m_fence;
    vk::Queue m_targetQueue;
    std::mutex *m_queueMutex = nullptr;
    vk::PipelineStageFlags m_waitStage;
    bool m_recording = false;
    bool m_begun = false;
    bool m_submitted = false;
    const GpuComputePipeline *m_boundPipeline = nullptr;
    std::vector<StagingBuffer> m_staging;
    std::vector<PendingDownload> m_pendingDownloads;
};

/// Submit a compute dispatch in a single call (begin, bind, dispatch, end, submitAndWait).
void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline, uint32_t groupsX,
                     uint32_t groupsY = 1, uint32_t groupsZ = 1);

/// Submit a compute dispatch with push constants.
void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline, const void *pushConstantData,
                     size_t pushConstantSize, uint32_t groupsX, uint32_t groupsY = 1,
                     uint32_t groupsZ = 1);

template <typename T>
void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline, const T &pushConstants,
                     uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1)
{
    oneShotDispatch(ctx, pipeline, &pushConstants, sizeof(T), groupsX, groupsY, groupsZ);
}

/// Submit a compute dispatch using the pipeline's default group counts.
void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline);

} // namespace pix
