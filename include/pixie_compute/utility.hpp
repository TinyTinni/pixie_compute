#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace pix
{

namespace detail
{

/// Convert a contiguous range of trivially copyable elements to a byte span.
template <std::ranges::contiguous_range R>
    requires std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
std::span<const std::byte> asByteSpan(const R &range) noexcept
{
    return {reinterpret_cast<const std::byte *>(std::data(range)),
            std::size(range) * sizeof(std::ranges::range_value_t<R>)};
}

/// Convert a contiguous range of non-const, trivially copyable elements to a
/// writable byte span.
template <std::ranges::contiguous_range R>
    requires(std::is_trivially_copyable_v<std::ranges::range_value_t<R>> &&
             !std::is_const_v<std::ranges::range_value_t<R>>)
std::span<std::byte> asWritableByteSpan(R &range) noexcept
{
    return {reinterpret_cast<std::byte *>(std::data(range)),
            std::size(range) * sizeof(std::ranges::range_value_t<R>)};
}

} // namespace detail

/// Exception thrown on GPU or shader compilation errors.
/// All library failures (Vulkan API errors, VMA failures, shader compilation errors,
/// invalid usage) throw this type. Vulkan errors are wrapped with context messages.
struct GpuError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/// Ceiling integer division for workgroup count calculation (1D).
constexpr uint32_t workgroupCount(uint32_t dim, uint32_t localSize = 8)
{
    return (dim + localSize - 1) / localSize;
}

/// Ceiling integer division for workgroup count calculation (2D).
constexpr std::array<uint32_t, 2> workgroupCount2D(uint32_t dimX, uint32_t dimY,
                                                   uint32_t localX = 8, uint32_t localY = 8)
{
    return {workgroupCount(dimX, localX), workgroupCount(dimY, localY)};
}

/// Ceiling integer division for workgroup count calculation (3D).
constexpr std::array<uint32_t, 3> workgroupCount3D(uint32_t dimX, uint32_t dimY, uint32_t dimZ,
                                                   uint32_t localX = 8, uint32_t localY = 8,
                                                   uint32_t localZ = 8)
{
    return {workgroupCount(dimX, localX), workgroupCount(dimY, localY),
            workgroupCount(dimZ, localZ)};
}

} // namespace pix
