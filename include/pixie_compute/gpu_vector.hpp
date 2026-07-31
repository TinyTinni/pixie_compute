#pragma once

#include "pixie_compute/gpu_buffer.hpp"

#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace pix
{

/// Typed, resizable vector storage backed by a GpuBuffer.
///
/// GpuVector owns its buffer and uses element counts rather than byte sizes at
/// call sites. Resizing preserves existing elements and may block while a
/// device-local buffer is copied through staging memory.
template <typename T> class GpuVector
{
    static_assert(std::is_trivially_copyable_v<T>, "GpuVector elements must be trivially copyable");

    public:
    using value_type = T;
    using BufferType = GpuBuffer::Type;

    explicit GpuVector(GpuContext &ctx, size_t count = 0, BufferType type = BufferType::Device,
                       vk::BufferUsageFlags extraUsage = {})
        : m_ctx(&ctx), m_type(type), m_extraUsage(extraUsage)
    {
        resize(count);
    }

    ~GpuVector() = default;

    GpuVector(const GpuVector &) = delete;
    GpuVector &operator=(const GpuVector &) = delete;
    GpuVector(GpuVector &&) noexcept = default;
    GpuVector &operator=(GpuVector &&) noexcept = default;

    size_t size() const noexcept { return m_size; }
    size_t size_bytes() const noexcept { return m_size * sizeof(T); }
    bool empty() const noexcept { return m_size == 0; }
    BufferType type() const noexcept { return m_type; }

    /// Resize while preserving the prefix shared by the old and new vectors.
    void resize(size_t count)
    {
        if (count == m_size)
            return;

        if (count == 0)
        {
            m_buffer.reset();
            m_size = 0;
            return;
        }

        auto replacement =
            std::make_unique<GpuBuffer>(*m_ctx, count * sizeof(T), m_type, m_extraUsage);
        if (m_buffer && m_size != 0)
        {
            const size_t preserved = (m_size < count) ? m_size : count;
            std::vector<T> values(preserved);
            m_buffer->download(values);
            replacement->upload(std::span<const T>(values));
        }
        m_buffer = std::move(replacement);
        m_size = count;
    }

    /// Replace all elements and resize to match the input.
    void assign(std::span<const T> values)
    {
        resize(values.size());
        if (!values.empty())
            m_buffer->upload(values);
    }

    template <std::ranges::contiguous_range R>
    requires std::same_as<std::ranges::range_value_t<R>, T> void assign(const R &values)
    {
        assign(std::span<const T>(values));
    }

    void upload(std::span<const T> values, size_t elementOffset = 0)
    {
        checkRange(elementOffset, values.size());
        if (!values.empty())
            m_buffer->upload(values, elementOffset * sizeof(T));
    }

    template <std::ranges::contiguous_range R>
    requires std::same_as<std::ranges::range_value_t<R>, T> void upload(const R &values,
                                                                        size_t elementOffset = 0)
    {
        upload(std::span<const T>(values), elementOffset);
    }

    void download(std::span<T> values, size_t elementOffset = 0) const
    {
        checkRange(elementOffset, values.size());
        if (!values.empty())
            m_buffer->download(values, elementOffset * sizeof(T));
    }

    template <std::ranges::contiguous_range R>
    requires(!std::is_const_v<std::ranges::range_value_t<R>> &&
             std::same_as<std::ranges::range_value_t<R>, T>) void download(R &values,
                                                                           size_t elementOffset =
                                                                               0) const
    {
        download(std::span<T>(values), elementOffset);
    }

    std::span<T> mapped()
    {
        if (!m_buffer)
            return {};
        return m_buffer->template mapped<T>();
    }

    std::span<const T> mapped() const
    {
        if (!m_buffer)
            return {};
        return m_buffer->template mapped<T>();
    }

    void clear(uint32_t value = 0)
    {
        if (m_buffer)
            m_buffer->clear(value);
    }

    GpuBuffer &buffer()
    {
        if (!m_buffer)
            throw GpuError("operation on empty GpuVector");
        return *m_buffer;
    }

    const GpuBuffer &buffer() const
    {
        if (!m_buffer)
            throw GpuError("operation on empty GpuVector");
        return *m_buffer;
    }

    vk::Buffer handle() const { return buffer().handle(); }

    vk::DescriptorBufferInfo descriptorInfo(vk::DeviceSize elementOffset = 0,
                                            vk::DeviceSize elementCount = VK_WHOLE_SIZE) const
    {
        if (elementOffset > m_size)
            throw GpuError("GpuVector descriptor offset exceeds vector size");
        if (elementCount == VK_WHOLE_SIZE)
            elementCount = m_size - elementOffset;
        if (elementCount > m_size - elementOffset)
            throw GpuError("GpuVector descriptor range exceeds vector size");
        return buffer().descriptorInfo(elementOffset * sizeof(T), elementCount * sizeof(T));
    }

    private:
    void checkRange(size_t offset, size_t count) const
    {
        if (!m_buffer || offset > m_size || count > m_size - offset)
            throw GpuError("GpuVector range exceeds vector size");
    }

    GpuContext *m_ctx;
    BufferType m_type;
    vk::BufferUsageFlags m_extraUsage;
    std::unique_ptr<GpuBuffer> m_buffer;
    size_t m_size = 0;
};

} // namespace pix
