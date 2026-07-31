#pragma once

#include "pixie_compute/detail/vulkan_include.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>

struct VmaAllocator_T;
struct VmaAllocation_T;

namespace pix
{

class GpuContext;
class GpuCommandBuffer;

/// Configuration for GpuImage construction.
struct GpuImageDesc
{
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;
    vk::Format format = vk::Format::eUndefined;
    vk::ImageUsageFlags usage = {};
    vk::ImageType imageType = vk::ImageType::e2D;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    vk::ImageCreateFlags flags = {};
    vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
};

/// RAII wrapper for a Vulkan image, image view, and optional VMA allocation.
///
/// For linear, typeless GPU data (arrays, structs, particle data, I/O), use GpuBuffer instead.
///
/// Shader resource mapping:
///   ┌──────────────┬──────────────────────┬──────────────────────┬─────────────────────┐
///   │ C++ class    │ HLSL type            │ Descriptor type      │ Access pattern      │
///   ├──────────────┼──────────────────────┼──────────────────────┼─────────────────────┤
///   │ GpuBuffer    │ StructuredBuffer<T>  │ eStorageBuffer       │ index [i]           │
///   │ GpuBuffer    │ RWStructuredBuffer<T>│ eStorageBuffer       │ index [i]           │
///   │ GpuImage     │ Texture2D<T>         │ eSampledImage        │ coords (x,y)        │
///   │ GpuImage     │ RWTexture2D<T>       │ eStorageImage        │ coords (x,y)        │
///   │ GpuImage     │ Texture2DArray<T>    │ eSampledImage        │ coords (x,y,layer)  │
///   │ GpuImage     │ RWTexture2DArray<T>  │ eStorageImage        │ coords (x,y,layer)  │
///   └──────────────┴──────────────────────┴──────────────────────┴─────────────────────┘
///
/// Layout tracking:
///   The image tracks its current layout internally. upload() transitions from
///   eUndefined to the specified finalLayout. Use transition() for manual layout
///   changes (e.g. between compute passes). descriptorInfo() defaults to the
///   tracked layout.
///
/// Tiling:
///   eOptimal (default): GPU-friendly swizzled layout. Best for shader access.
///                       Requires staging buffer for CPU↔GPU transfer.
///   eLinear:            Texels laid out row-by-row. Supports host mapping
///                       for direct CPU read/write via vkMapMemory.
///                       May have worse GPU cache locality and filter rate.
///                       Check vkGetPhysicalDeviceFormatProperties for support.
///
/// @see GpuBuffer
class GpuImage
{
    friend class GpuCommandBuffer;

    public:
    GpuImage() = default;

    /// Create a VMA-backed VkImage + VkImageView.
    GpuImage(GpuContext &ctx, const GpuImageDesc &desc);

    /// Adopt an existing VkImage (e.g. from external memory). No VMA allocation.
    GpuImage(GpuContext &ctx, vk::Image image, vk::Format format, uint32_t width, uint32_t height,
             vk::ImageViewType viewType = vk::ImageViewType::e2D, uint32_t mipLevels = 1,
             uint32_t arrayLayers = 1, uint32_t depth = 1);

    ~GpuImage();

    GpuImage(const GpuImage &) = delete;
    GpuImage &operator=(const GpuImage &) = delete;
    GpuImage(GpuImage &&other) noexcept;
    GpuImage &operator=(GpuImage &&other) noexcept;

    vk::Image image() const noexcept { return *m_image; }
    vk::ImageView imageView() const noexcept { return *m_imageView; }
    vk::Format format() const noexcept { return m_format; }
    uint32_t width() const noexcept { return m_width; }
    uint32_t height() const noexcept { return m_height; }
    uint32_t depth() const noexcept { return m_depth; }
    uint32_t mipLevels() const noexcept { return m_mipLevels; }
    uint32_t arrayLayers() const noexcept { return m_arrayLayers; }
    vk::ImageType imageType() const noexcept { return m_imageType; }
    vk::ImageLayout currentLayout() const noexcept { return m_layout; }

    /// Descriptor info for binding this image to a shader.
    /// Uses the tracked layout and no sampler.
    vk::DescriptorImageInfo descriptorInfo() const;
    /// Descriptor info with a sampler (for combined image samplers).
    vk::DescriptorImageInfo descriptorInfo(vk::Sampler sampler) const;
    /// Descriptor info with explicit layout override.
    vk::DescriptorImageInfo descriptorInfo(vk::ImageLayout layout, vk::Sampler sampler = {}) const;

    /// Upload pixel data to a subresource via staging buffer.
    ///
    /// Layout transitions (automatic):
    ///   eUndefined (input) → eTransferDstOptimal → finalLayout (output)
    ///
    /// After upload the subresource is in @p finalLayout, ready for shader access.
    /// Only valid for freshly-created or reinitialized subresources (discards old
    /// contents). For incremental updates, use GpuCommandBuffer directly.
    ///
    /// @param data        Source pixel data (tightly packed, single subresource).
    /// @param size        Size of source data in bytes.
    /// @param finalLayout Desired layout after the upload completes.
    /// @param mipLevel    Target mip level (width/height/depth are divided by 2^level).
    /// @param arrayLayer  Target array layer.
    void upload(const void *data, size_t size,
                vk::ImageLayout finalLayout = vk::ImageLayout::eGeneral, uint32_t mipLevel = 0,
                uint32_t arrayLayer = 0);

    void upload(std::span<const std::byte> data,
                vk::ImageLayout finalLayout = vk::ImageLayout::eGeneral, uint32_t mipLevel = 0,
                uint32_t arrayLayer = 0)
    {
        upload(data.data(), data.size(), finalLayout, mipLevel, arrayLayer);
    }

    template <typename T>
    void upload(std::span<const T> data, vk::ImageLayout finalLayout = vk::ImageLayout::eGeneral,
                uint32_t mipLevel = 0, uint32_t arrayLayer = 0)
    {
        upload(std::as_bytes(data), finalLayout, mipLevel, arrayLayer);
    }

    template <std::ranges::contiguous_range R>
        requires std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
    void upload(const R &data, vk::ImageLayout finalLayout = vk::ImageLayout::eGeneral,
                uint32_t mipLevel = 0, uint32_t arrayLayer = 0)
    {
        upload(std::data(data), std::size(data) * sizeof(std::ranges::range_value_t<R>),
               finalLayout, mipLevel, arrayLayer);
    }

    /// Download pixel data from a subresource via staging buffer.
    ///
    /// Layout transitions (automatic):
    ///   currentLayout → eTransferSrcOptimal → currentLayout (restored)
    ///
    /// @param data       Destination buffer for pixel data.
    /// @param size       Size of destination buffer in bytes.
    /// @param mipLevel   Source mip level.
    /// @param arrayLayer Source array layer.
    void download(void *data, size_t size, uint32_t mipLevel = 0, uint32_t arrayLayer = 0);

    /// Transition the image to a new layout, recording a pipeline barrier.
    /// Updates the tracked layout.
    /// @param cmd        Command buffer to record the barrier into.
    /// @param newLayout  Target layout.
    /// @param srcStage   Source pipeline stage (default: eAllCommands).
    /// @param srcAccess  Source access flags (default: shader read/write).
    /// @param dstStage   Destination pipeline stage (default: eAllCommands).
    /// @param dstAccess  Destination access flags (default: shader read/write).
    void transition(GpuCommandBuffer &cmd, vk::ImageLayout newLayout,
                    vk::PipelineStageFlags srcStage = vk::PipelineStageFlagBits::eAllCommands,
                    vk::AccessFlags srcAccess = vk::AccessFlagBits::eShaderRead |
                                                vk::AccessFlagBits::eShaderWrite,
                    vk::PipelineStageFlags dstStage = vk::PipelineStageFlagBits::eAllCommands,
                    vk::AccessFlags dstAccess = vk::AccessFlagBits::eShaderRead |
                                                vk::AccessFlagBits::eShaderWrite);

    /// Assign a debug name to the image (VK_EXT_debug_utils). No-op when the
    /// extension is unavailable.
    void setDebugName(const std::string &name);

    private:
    void destroy() noexcept;
    void swap(GpuImage &other) noexcept;

    vk::ImageViewType deduceViewType(vk::ImageType imageType, uint32_t arrayLayers) const;
    size_t subresourceFootprint(uint32_t mipLevel) const;

    GpuContext *m_ctx = nullptr;
    vk::UniqueImage m_image;
    vk::UniqueImageView m_imageView;
    VmaAllocation_T *m_allocation = nullptr;
    vk::Format m_format = vk::Format::eUndefined;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_depth = 1;
    uint32_t m_mipLevels = 1;
    uint32_t m_arrayLayers = 1;
    vk::ImageType m_imageType = vk::ImageType::e2D;
    bool m_externalImage = false;
    vk::ImageLayout m_layout = vk::ImageLayout::eUndefined;
};

} // namespace pix
