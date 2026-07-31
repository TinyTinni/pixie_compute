#pragma once

#include "pixie_compute/gpu_vector.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <span>
#include <type_traits>

namespace pix
{

template <typename T, size_t Rank> class GpuTensor;

template <typename T, size_t Rank> class GpuTensorView
{
    static_assert(Rank > 0, "GpuTensor rank must be positive");
    static_assert(std::is_trivially_copyable_v<T>, "GpuTensor elements must be trivially copyable");

    public:
    using Shape = std::array<size_t, Rank>;

    GpuTensorView() = default;

    size_t elements() const noexcept { return m_elements; }
    size_t size_bytes() const noexcept { return m_elements * sizeof(T); }
    const Shape &shape() const noexcept { return m_shape; }
    const Shape &strides() const noexcept { return m_strides; }
    bool empty() const noexcept { return m_elements == 0; }

    vk::DescriptorBufferInfo descriptorInfo() const
    {
        if (!m_buffer)
            throw GpuError("descriptorInfo on an empty GpuTensorView");
        return m_buffer->descriptorInfo(m_offset * sizeof(T), m_elements * sizeof(T));
    }

    GpuBuffer &buffer() const
    {
        if (!m_buffer)
            throw GpuError("buffer access on an empty GpuTensorView");
        return *m_buffer;
    }

    /// Create a subview starting at `offset` with the given `shape`, both in
    /// elements. The subview reuses the parent's strides, so it stays a view
    /// into the same storage. Throws if the region exceeds the view bounds.
    GpuTensorView subview(const Shape &offset, const Shape &shape) const
    {
        if (!m_buffer)
            throw GpuError("subview on an empty GpuTensorView");
        size_t subOffset = m_offset;
        for (size_t i = 0; i < Rank; ++i)
        {
            if (offset[i] > m_shape[i] || shape[i] > m_shape[i] - offset[i])
                throw GpuError("GpuTensorView subview exceeds view bounds");
            if (offset[i] != 0)
                subOffset += offset[i] * m_strides[i];
        }
        return GpuTensorView(*m_buffer, shape, m_strides, subOffset);
    }

    /// Create a subview starting at `offset`, extending to the parent's edge.
    GpuTensorView subview(const Shape &offset) const
    {
        if (!m_buffer)
            throw GpuError("subview on an empty GpuTensorView");
        Shape shape = m_shape;
        for (size_t i = 0; i < Rank; ++i)
        {
            if (offset[i] > shape[i])
                throw GpuError("GpuTensorView subview exceeds view bounds");
            shape[i] -= offset[i];
        }
        return subview(offset, shape);
    }

    private:
    friend class GpuTensor<T, Rank>;

    GpuTensorView(GpuBuffer &buffer, Shape shape, Shape strides, size_t offset)
        : m_buffer(&buffer), m_shape(shape), m_strides(strides), m_offset(offset),
          m_elements(product(shape))
    {
    }

    static size_t product(const Shape &shape)
    {
        size_t result = 1;
        for (size_t value : shape)
        {
            if (value != 0 && result > std::numeric_limits<size_t>::max() / value)
                throw GpuError("GpuTensor shape overflows size_t");
            result *= value;
        }
        return result;
    }

    GpuBuffer *m_buffer = nullptr;
    Shape m_shape{};
    Shape m_strides{};
    size_t m_offset = 0;
    size_t m_elements = 0;
};

template <typename T, size_t Rank> class GpuTensor
{
    static_assert(Rank > 0, "GpuTensor rank must be positive");
    static_assert(std::is_trivially_copyable_v<T>, "GpuTensor elements must be trivially copyable");

    public:
    using Shape = std::array<size_t, Rank>;
    using View = GpuTensorView<T, Rank>;
    using BufferType = GpuBuffer::Type;

    GpuTensor(GpuContext &ctx, Shape shape, BufferType type = BufferType::Device,
              vk::BufferUsageFlags extraUsage = {})
        : m_shape(shape), m_strides(makeStrides(shape)),
          m_storage(ctx, product(shape), type, extraUsage)
    {
    }

    size_t elements() const noexcept { return m_storage.size(); }
    size_t size_bytes() const noexcept { return m_storage.size_bytes(); }
    bool empty() const noexcept { return m_storage.empty(); }
    const Shape &shape() const noexcept { return m_shape; }
    const Shape &strides() const noexcept { return m_strides; }

    void upload(std::span<const T> values)
    {
        if (values.size() != elements())
            throw GpuError("GpuTensor upload size does not match shape");
        m_storage.upload(values);
    }

    template <std::ranges::contiguous_range R>
        requires std::same_as<std::ranges::range_value_t<R>, T>
    void upload(const R &values)
    {
        upload(std::span<const T>(values));
    }

    void download(std::span<T> values) const
    {
        if (values.size() != elements())
            throw GpuError("GpuTensor download size does not match shape");
        m_storage.download(values);
    }

    template <std::ranges::contiguous_range R>
        requires std::same_as<std::ranges::range_value_t<R>, T>
    void download(R &values) const
    {
        download(std::span<T>(values));
    }

    std::span<T> mapped() { return m_storage.mapped(); }
    std::span<const T> mapped() const { return m_storage.mapped(); }
    void clear(uint32_t value = 0) { m_storage.clear(value); }

    GpuBuffer &buffer() { return m_storage.buffer(); }
    const GpuBuffer &buffer() const { return m_storage.buffer(); }
    vk::Buffer handle() const { return m_storage.handle(); }
    vk::DescriptorBufferInfo descriptorInfo() const { return m_storage.descriptorInfo(); }

    View view() { return View(m_storage.buffer(), m_shape, m_strides, 0); }
    View reshape(Shape shape)
    {
        if (product(shape) != elements())
            throw GpuError("GpuTensor reshape changes element count");
        return View(m_storage.buffer(), shape, makeStrides(shape), 0);
    }

    private:
    static size_t product(const Shape &shape)
    {
        size_t result = 1;
        for (size_t value : shape)
        {
            if (value != 0 && result > std::numeric_limits<size_t>::max() / value)
                throw GpuError("GpuTensor shape overflows size_t");
            result *= value;
        }
        return result;
    }

    static Shape makeStrides(const Shape &shape)
    {
        Shape result{};
        result[Rank - 1] = 1;
        for (size_t i = Rank - 1; i > 0; --i)
            result[i - 1] = result[i] * shape[i];
        return result;
    }

    Shape m_shape;
    Shape m_strides;
    GpuVector<T> m_storage;
};

} // namespace pix
