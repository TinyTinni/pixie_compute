#pragma once

#include "pixie_compute/detail/vulkan_include.hpp"
#include "pixie_compute/utility.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace pix
{

class GpuContext;
class GpuComputePipeline;
class GpuBuffer;
class GpuTimelineSemaphore;
struct StagingBuffer;

/// Deferred result for a GPU-to-CPU readback.
///
/// The handle owns the submission and staging memory. Calling wait() or get()
/// blocks until the copy completes; destroying the handle also waits safely.
class GpuDownload
{
    public:
    GpuDownload() = default;
    ~GpuDownload();

    GpuDownload(const GpuDownload &) = delete;
    GpuDownload &operator=(const GpuDownload &) = delete;
    GpuDownload(GpuDownload &&) noexcept;
    GpuDownload &operator=(GpuDownload &&) noexcept;

    void wait();
    std::vector<std::byte> get();
    const std::vector<std::byte> &bytes();
    template <typename T> std::vector<T> getAs()
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "GpuDownload elements must be trivially copyable");
        const auto &raw = bytes();
        if (raw.size() % sizeof(T) != 0)
            throw GpuError("GpuDownload size is not a multiple of the requested type");
        std::vector<T> result(raw.size() / sizeof(T));
        std::memcpy(result.data(), raw.data(), raw.size());
        return result;
    }
    bool valid() const noexcept { return static_cast<bool>(m_state); }

    private:
    struct State;
    explicit GpuDownload(std::shared_ptr<State> state) : m_state(std::move(state)) {}
    friend class GpuBuffer;
    std::shared_ptr<State> m_state;
};

/// A single wait condition for a submission: a binary semaphore or a timeline
/// semaphore at a specific counter value.
///
/// The wait stage is the first synchronization scope stage — the earliest
/// pipeline stage of the submission that the wait gates. Defaults to
/// eAllCommands (safe for both compute and transfer queues); narrow it only
/// when the signaling submission performs work in a known subset of stages.
///
/// Use the static factories to construct; one of `binary` or `timeline` is set.
struct GpuWaitSemaphore
{
    /// Binary semaphore to wait on. Empty when `timeline` is set.
    vk::Semaphore binary = {};
    /// Timeline semaphore to wait on. Null when `binary` is set.
    const GpuTimelineSemaphore *timeline = nullptr;
    /// Timeline counter value to wait for (when `timeline` is set).
    uint64_t value = 0;
    /// Pipeline stage at which to wait.
    vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands;

    static GpuWaitSemaphore
    makeBinary(vk::Semaphore semaphore,
               vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands)
    {
        return {semaphore, nullptr, 0, stage};
    }

    static GpuWaitSemaphore
    makeTimeline(const GpuTimelineSemaphore &semaphore, uint64_t value,
                 vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands)
    {
        return {{}, &semaphore, value, stage};
    }
};

/// A single signal condition for a submission: a binary semaphore or a timeline
/// semaphore signaled to a specific counter value.
struct GpuSignalSemaphore
{
    /// Binary semaphore to signal. Empty when `timeline` is set.
    vk::Semaphore binary = {};
    /// Timeline semaphore to signal. Null when `binary` is set.
    const GpuTimelineSemaphore *timeline = nullptr;
    /// Timeline counter value to signal (when `timeline` is set).
    uint64_t value = 0;

    static GpuSignalSemaphore makeBinary(vk::Semaphore semaphore)
    {
        return {semaphore, nullptr, 0};
    }

    static GpuSignalSemaphore makeTimeline(const GpuTimelineSemaphore &semaphore, uint64_t value)
    {
        return {{}, &semaphore, value};
    }
};

/// Synchronization parameters for command buffer submission.
///
/// Holds any number of waits (applied before the submission executes) and
/// signals (applied after it completes), each either binary or timeline.
/// An empty GpuSubmitSync{} performs no synchronization.
///
/// Usage with a timeline semaphore:
///   uint64_t val = sem.next();
///   uploadCmd.submit({.signals = {GpuSignalSemaphore::makeTimeline(sem, val)}});
///   computeCmd.submitAndWait({.waits = {GpuWaitSemaphore::makeTimeline(sem, val)}});
///
/// Waits and signals must stay alive until the submission completes. Timeline
/// semaphores passed to submit() (non-blocking) must outlive wait().
struct GpuSubmitSync
{
    /// Semaphores to wait on before execution begins.
    std::vector<GpuWaitSemaphore> waits = {};
    /// Semaphores to signal after execution completes.
    std::vector<GpuSignalSemaphore> signals = {};
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

    /// Submit without blocking. Stores fence internally for wait().
    void submit(const GpuSubmitSync &sync = {});

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

    template <std::ranges::contiguous_range R>
        requires std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
    void upload(GpuBuffer &buf, const R &data, size_t byteOffset = 0)
    {
        const auto bytes = detail::asByteSpan(data);
        upload(buf, bytes.data(), bytes.size(), byteOffset);
    }

    template <std::ranges::contiguous_range R>
        requires(std::is_trivially_copyable_v<std::ranges::range_value_t<R>> &&
                 !std::is_const_v<std::ranges::range_value_t<R>>)
    void download(GpuBuffer &buf, R &data, size_t byteOffset = 0)
    {
        const auto bytes = detail::asWritableByteSpan(data);
        download(buf, bytes.data(), bytes.size(), byteOffset);
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
    /// Bind a specific descriptor set instance (see addBindingSet) of the pipeline.
    void bind(const GpuComputePipeline &pipeline, uint32_t bindingSetIndex);
    /// Push constants to the bound pipeline.
    void pushConstants(const void *data, size_t size);

    template <typename T> void pushConstants(const T &pc) { pushConstants(&pc, sizeof(T)); }

    /// Dispatch using the bound pipeline's default group counts.
    void dispatch();
    /// Dispatch with explicit workgroup counts.
    void dispatch(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1);

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
    bool m_recording = false;
    bool m_begun = false;
    bool m_submitted = false;
    const GpuComputePipeline *m_boundPipeline = nullptr;
    std::vector<StagingBuffer> m_staging;
    std::vector<PendingDownload> m_pendingDownloads;
    std::vector<std::string> m_recordedLabels;
    std::vector<std::string> m_labelStack;
};

/// Submit a compute dispatch in a single call (begin, bind, dispatch, end, submitAndWait).
void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline, uint32_t groupsX,
                     uint32_t groupsY = 1, uint32_t groupsZ = 1);

/// Submit a compute dispatch with push constants.
void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline, const void *pushConstantData,
                     size_t pushConstantSize, uint32_t groupsX, uint32_t groupsY = 1,
                     uint32_t groupsZ = 1);

namespace detail
{

/// Dispatch with push constants using the pipeline's default group counts.
/// Defined in gpu_command_buffer.cpp where GpuComputePipeline is complete.
void oneShotDispatchWithDefaultGroups(GpuContext &ctx, GpuComputePipeline &pipeline,
                                      const void *pushConstantData, size_t pushConstantSize);

} // namespace detail

template <typename T>
void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline, const T &pushConstants,
                     uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1)
{
    oneShotDispatch(ctx, pipeline, &pushConstants, sizeof(T), groupsX, groupsY, groupsZ);
}

/// Submit a compute dispatch with push constants using the pipeline's default
/// group counts.
template <typename T>
void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline, const T &pushConstants)
{
    detail::oneShotDispatchWithDefaultGroups(ctx, pipeline, &pushConstants, sizeof(T));
}

/// Submit a compute dispatch using the pipeline's default group counts.
void oneShotDispatch(GpuContext &ctx, GpuComputePipeline &pipeline);

} // namespace pix
