#pragma once

#include "pixie_compute/gpu_command_buffer.hpp"

#include <array>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace pix
{

class GpuTimelineSemaphore;
class GpuComputePipeline;

/// A timeline-backed dependency produced by a GpuStream submission.
class GpuEvent
{
    public:
    GpuEvent() = default;
    bool valid() const noexcept { return static_cast<bool>(m_state); }
    uint64_t value() const noexcept { return m_state ? m_state->value : 0; }

    private:
    struct State
    {
        std::shared_ptr<GpuTimelineSemaphore> semaphore;
        uint64_t value = 0;
    };

    explicit GpuEvent(std::shared_ptr<State> state) : m_state(std::move(state)) {}
    friend class GpuStream;
    std::shared_ptr<State> m_state;
};

/// Convenience scheduler for ordered compute or transfer work.
///
/// Work is recorded into a command buffer and submitted with commit(). Each
/// commit signals the stream timeline. Other streams can wait on the returned
/// event without handling raw semaphore values.
class GpuStream
{
    public:
    explicit GpuStream(GpuContext &ctx,
                       GpuCommandBuffer::QueueType type = GpuCommandBuffer::QueueType::Compute);
    ~GpuStream();

    GpuStream(const GpuStream &) = delete;
    GpuStream &operator=(const GpuStream &) = delete;
    GpuStream(GpuStream &&) = delete;
    GpuStream &operator=(GpuStream &&) = delete;

    void begin();
    GpuEvent commit();
    /// Wait for all committed work on the host. Pending recorded work is committed first.
    void wait();
    /// Make the next commit wait for an event from this or another stream.
    void wait(const GpuEvent &event);

    void upload(GpuBuffer &buffer, const void *data, size_t size, size_t offset = 0);
    void download(GpuBuffer &buffer, void *data, size_t size, size_t offset = 0);
    void copyBuffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size,
                    vk::DeviceSize srcOffset = 0, vk::DeviceSize dstOffset = 0);
    void fillBuffer(vk::Buffer buffer, uint32_t value, vk::DeviceSize size,
                    vk::DeviceSize offset = 0);
    void bufferBarrier(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
                       vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccess,
                       vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess);
    void imageBarrier(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                      vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccess,
                      vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess,
                      vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor,
                      uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED,
                      uint32_t dstQueueFamily = VK_QUEUE_FAMILY_IGNORED);

    void bind(const GpuComputePipeline &pipeline);
    void bind(const GpuComputePipeline &pipeline, uint32_t bindingSetIndex);
    void pushConstants(const void *data, size_t size);

    template <typename T> void pushConstants(const T &value) { pushConstants(&value, sizeof(T)); }

    void dispatch(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1);
    void dispatch();

    /// Byte-offset overloads, mirroring GpuCommandBuffer/GpuBuffer (offsets are
    /// in bytes, not elements). Use GpuVector/GpuTensor for element-typed access.
    template <std::ranges::contiguous_range R>
        requires std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
    void upload(GpuBuffer &buffer, const R &values, size_t byteOffset = 0)
    {
        const auto bytes = detail::asByteSpan(values);
        upload(buffer, bytes.data(), bytes.size(), byteOffset);
    }

    template <std::ranges::contiguous_range R>
        requires(std::is_trivially_copyable_v<std::ranges::range_value_t<R>> &&
                 !std::is_const_v<std::ranges::range_value_t<R>>)
    void download(GpuBuffer &buffer, R &values, size_t byteOffset = 0)
    {
        const auto bytes = detail::asWritableByteSpan(values);
        download(buffer, bytes.data(), bytes.size(), byteOffset);
    }

    template <typename T>
    void download(GpuBuffer &buffer, std::span<T> values, size_t byteOffset = 0)
    {
        download(buffer, values.data(), values.size_bytes(), byteOffset);
    }

    bool recording() const noexcept;
    GpuCommandBuffer &commandBuffer();

    private:
    struct InFlight
    {
        std::unique_ptr<GpuCommandBuffer> command;
        uint64_t value = 0;
    };

    void ensureRecording();
    void releaseCompletedSubmissions();

    GpuContext *m_ctx;
    GpuCommandBuffer::QueueType m_type;
    std::shared_ptr<GpuTimelineSemaphore> m_timeline;
    std::unique_ptr<GpuCommandBuffer> m_current;
    std::vector<InFlight> m_inFlight;
    std::vector<GpuEvent> m_pendingEvents;
};

} // namespace pix
