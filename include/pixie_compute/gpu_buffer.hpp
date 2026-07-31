#pragma once

#include "pixie_compute/utility.hpp"

#include <cstdint>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>
#include <vulkan/vulkan.hpp>

struct VmaAllocation_T;

namespace pix
{

class GpuContext;
class GpuCommandBuffer;

/// A pooled staging buffer managed by GpuContext. Internal use; acquire/release
/// are private methods of GpuContext.
struct StagingBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation_T *allocation = nullptr;
    void *mapped = nullptr;
    size_t size = 0;
};

/// GPU buffer backed by VMA, supporting device-local and host-coherent memory.
///
/// For 2D/3D texel grids with GPU-optimal tiling, format-aware access, mipmaps,
/// texture arrays, or external image import, use GpuImage instead.
/// @see GpuImage
class GpuBuffer
{
    friend class GpuCommandBuffer;

    public:
    enum class Type
    {
        /// GPU-local memory. Fastest for GPU compute workloads.
        /// Not directly CPU-mappable; upload/download go through a staging buffer
        /// automatically. Best for large buffers used primarily by compute shaders,
        /// write-once-then-read patterns, and maximum GPU throughput.
        Device,
        /// Host-visible, host-coherent memory (CPU-mappable via persistent pointer).
        /// Simpler API: upload/download are direct memcpy, no GPU submission needed.
        /// Best for small buffers, frequent CPU readback, prototyping, or when you
        /// need mapped<T>() direct access. May have lower GPU bandwidth than Device
        /// memory on discrete GPUs.
        HostCoherent
    };

    /// Create a GPU buffer.
    /// @param size        Buffer size in bytes.
    /// @param type        Memory type (Device or HostCoherent).
    /// @param extraUsage  Additional Vulkan usage flags (e.g. eVertexBuffer).
    GpuBuffer(GpuContext &ctx, size_t size, Type type, vk::BufferUsageFlags extraUsage = {});
    ~GpuBuffer();

    GpuBuffer(const GpuBuffer &) = delete;
    GpuBuffer &operator=(const GpuBuffer &) = delete;
    GpuBuffer(GpuBuffer &&) noexcept;
    GpuBuffer &operator=(GpuBuffer &&) noexcept;

    size_t size() const noexcept { return m_size; }
    Type type() const noexcept { return m_type; }
    vk::Buffer handle() const noexcept { return m_buffer; }
    vk::DescriptorBufferInfo descriptorInfo(vk::DeviceSize offset = 0,
                                             vk::DeviceSize range = VK_WHOLE_SIZE) const;

    /// Returns a mutable span over the host-visible mapped memory.
    /// Returns an empty span for device-only buffers.
    template <typename T> std::span<T> mapped()
    {
        if (!m_mapped)
            return {};
        return {static_cast<T *>(m_mapped), m_size / sizeof(T)};
    }

    /// Returns a const span over the host-visible mapped memory.
    /// Returns an empty span for device-only buffers.
    template <typename T> std::span<const T> mapped() const
    {
        if (!m_mapped)
            return {};
        return {static_cast<const T *>(m_mapped), m_size / sizeof(T)};
    }

    /// Upload data to the buffer. Works for both Device and HostCoherent types.
    /// Device buffers stage through a shared staging buffer via a one-shot command.
    void upload(const void *data, size_t size, size_t offset = 0);
    /// Download data from the buffer. Works for both Device and HostCoherent types.
    /// Device buffers stage through a shared staging buffer via a one-shot command.
    void download(void *data, size_t size, size_t offset = 0);

    /// Upload from a contiguous range (vector, array, span, etc.).
    template <std::ranges::contiguous_range R>
        requires std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
    void upload(const R &data, size_t byteOffset = 0)
    {
        upload(std::data(data), std::size(data) * sizeof(std::ranges::range_value_t<R>), byteOffset);
    }

    /// Download into a contiguous range (vector, array, span, etc.).
    template <std::ranges::contiguous_range R>
        requires std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
    void download(R &data, size_t byteOffset = 0)
    {
        download(std::data(data), std::size(data) * sizeof(std::ranges::range_value_t<R>), byteOffset);
    }

    template <typename T> void download(std::span<T> data, size_t byteOffset = 0)
    {
        download(data.data(), data.size_bytes(), byteOffset);
    }

    /// Fill the buffer with a 32-bit pattern (default: zero).
    /// Device buffers: one-shot GPU fillBuffer command.
    /// HostCoherent buffers: direct CPU memset/fill.
    void clear(uint32_t value = 0);

    /// Assign a debug name to the buffer (VK_EXT_debug_utils). No-op when the
    /// extension is unavailable.
    void setDebugName(const std::string &name);

    private:
    void swap(GpuBuffer &other) noexcept;

    GpuContext *m_ctx = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation_T *m_allocation = nullptr;
    void *m_mapped = nullptr;
    size_t m_size = 0;
    Type m_type = Type::Device;
};

/// Lightweight view into a sub-range of a GpuBuffer.
/// Forwards offset-aware operations to the parent buffer.
/// Copyable non-owning view (like std::string_view); the user must ensure the
/// parent buffer outlives all slices.
class GpuBufferSlice
{
    GpuBuffer *m_buf{};
    size_t m_offset = 0;
    size_t m_size = 0;

    public:
    GpuBufferSlice() = default;
    GpuBufferSlice(GpuBuffer &buf, size_t offset, size_t size)
        : m_buf(&buf), m_offset(offset), m_size(size)
    {
    }

    GpuBufferSlice(const GpuBufferSlice &) = default;
    GpuBufferSlice &operator=(const GpuBufferSlice &) = default;
    GpuBufferSlice(GpuBufferSlice &&) noexcept = default;
    GpuBufferSlice &operator=(GpuBufferSlice &&) noexcept = default;

    /// Upload data into this slice's region.
    void upload(const void *data, size_t size, size_t uploadOffset = 0);
    /// Download data from this slice's region.
    void download(void *data, size_t size, size_t downloadOffset = 0);

    template <std::ranges::contiguous_range R>
        requires std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
    void upload(const R &data, size_t byteOffset = 0)
    {
        upload(std::data(data), std::size(data) * sizeof(std::ranges::range_value_t<R>), byteOffset);
    }

    template <std::ranges::contiguous_range R>
        requires std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
    void download(R &data, size_t byteOffset = 0)
    {
        download(std::data(data), std::size(data) * sizeof(std::ranges::range_value_t<R>), byteOffset);
    }

    template <typename T> void download(std::span<T> data, size_t byteOffset = 0)
    {
        download(data.data(), data.size_bytes(), byteOffset);
    }

    GpuBuffer &parent()
    {
        if (!m_buf)
            throw GpuError("operation on empty GpuBufferSlice");
        return *m_buf;
    }
    const GpuBuffer &parent() const
    {
        if (!m_buf)
            throw GpuError("operation on empty GpuBufferSlice");
        return *m_buf;
    }
    vk::Buffer handle() const
    {
        if (!m_buf)
            throw GpuError("operation on empty GpuBufferSlice");
        return m_buf->handle();
    }
    size_t size() const noexcept { return m_size; }
    size_t offset() const noexcept { return m_offset; }

    vk::DescriptorBufferInfo descriptorInfo(vk::DeviceSize descriptorOffset = 0,
                                             vk::DeviceSize range = VK_WHOLE_SIZE) const
    {
        if (!m_buf)
            throw GpuError("operation on empty GpuBufferSlice");
        if (descriptorOffset > m_size)
            throw GpuError("descriptorOffset exceeds slice size");
        if (range == VK_WHOLE_SIZE)
            range = m_size - descriptorOffset;
        return m_buf->descriptorInfo(m_offset + descriptorOffset, range);
    }

    /// Returns a mutable span over the slice's region in the parent's mapped memory.
    /// Returns an empty span if the parent is device-only or the slice is empty.
    template <typename T> std::span<T> mapped()
    {
        if (!m_buf || !m_size)
            return {};
        auto s = m_buf->mapped<T>();
        if (s.empty())
            return {};
        return s.subspan(m_offset / sizeof(T), m_size / sizeof(T));
    }

    template <typename T> std::span<const T> mapped() const
    {
        if (!m_buf || !m_size)
            return {};
        auto s = m_buf->mapped<T>();
        if (s.empty())
            return {};
        return s.subspan(m_offset / sizeof(T), m_size / sizeof(T));
    }
};

} // namespace pix
