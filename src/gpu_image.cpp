#include "pixie_compute/gpu_image.hpp"

#include "error_utils.hpp"
#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"
#include "staging_scope.hpp"

#include <cstring>
#include <limits>
#include <vk_mem_alloc.h>

namespace pix
{
namespace
{

size_t formatBytes(vk::Format format)
{
    switch (format)
    {
    case vk::Format::eR8Unorm:
    case vk::Format::eR8Snorm:
    case vk::Format::eR8Uscaled:
    case vk::Format::eR8Sscaled:
    case vk::Format::eR8Uint:
    case vk::Format::eR8Sint:
        return 1;
    case vk::Format::eR16Unorm:
    case vk::Format::eR16Snorm:
    case vk::Format::eR16Uscaled:
    case vk::Format::eR16Sscaled:
    case vk::Format::eR16Uint:
    case vk::Format::eR16Sint:
    case vk::Format::eR16Sfloat:
    case vk::Format::eR8G8Unorm:
    case vk::Format::eR8G8Snorm:
    case vk::Format::eR8G8Uscaled:
    case vk::Format::eR8G8Sscaled:
    case vk::Format::eR8G8Uint:
    case vk::Format::eR8G8Sint:
        return 2;
    case vk::Format::eR32Uint:
    case vk::Format::eR32Sint:
    case vk::Format::eR32Sfloat:
    case vk::Format::eR16G16Unorm:
    case vk::Format::eR16G16Snorm:
    case vk::Format::eR16G16Uscaled:
    case vk::Format::eR16G16Sscaled:
    case vk::Format::eR16G16Uint:
    case vk::Format::eR16G16Sint:
    case vk::Format::eR16G16Sfloat:
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Snorm:
    case vk::Format::eR8G8B8A8Uscaled:
    case vk::Format::eR8G8B8A8Sscaled:
    case vk::Format::eR8G8B8A8Uint:
    case vk::Format::eR8G8B8A8Sint:
    case vk::Format::eB8G8R8A8Unorm:
        return 4;
    case vk::Format::eR32G32Uint:
    case vk::Format::eR32G32Sint:
    case vk::Format::eR32G32Sfloat:
    case vk::Format::eR16G16B16A16Unorm:
    case vk::Format::eR16G16B16A16Snorm:
    case vk::Format::eR16G16B16A16Uscaled:
    case vk::Format::eR16G16B16A16Sscaled:
    case vk::Format::eR16G16B16A16Uint:
    case vk::Format::eR16G16B16A16Sint:
    case vk::Format::eR16G16B16A16Sfloat:
        return 8;
    case vk::Format::eR32G32B32A32Uint:
    case vk::Format::eR32G32B32A32Sint:
    case vk::Format::eR32G32B32A32Sfloat:
        return 16;
    default:
        throw GpuError("unsupported or compressed image format for tightly-packed transfers");
    }
}

uint32_t mipExtent(uint32_t value, uint32_t level)
{
    return std::max(1u, value >> level);
}

} // namespace

GpuImage::GpuImage(GpuContext &ctx, const GpuImageDesc &desc)
    : m_ctx(&ctx), m_format(desc.format), m_width(desc.width), m_height(desc.height),
      m_depth(desc.depth), m_mipLevels(desc.mipLevels), m_arrayLayers(desc.arrayLayers),
      m_imageType(desc.imageType), m_externalImage(false), m_layout(vk::ImageLayout::eUndefined)
{
    if (m_width == 0 || m_height == 0 || m_depth == 0 || m_mipLevels == 0 || m_arrayLayers == 0)
        throw GpuError("GpuImage dimensions, mipLevels, and arrayLayers must be non-zero");
    if (m_imageType != vk::ImageType::e3D && m_depth != 1)
        throw GpuError("depth is only valid for 3D images");

    vk::ImageCreateInfo imageInfo(
        desc.flags, desc.imageType, desc.format, vk::Extent3D(desc.width, desc.height, desc.depth),
        desc.mipLevels, desc.arrayLayers, vk::SampleCountFlagBits::e1, desc.tiling, desc.usage,
        vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined);

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkImage rawImage = VK_NULL_HANDLE;
    auto result =
        vmaCreateImage(ctx.allocator(), reinterpret_cast<const VkImageCreateInfo *>(&imageInfo),
                       &allocInfo, &rawImage, &m_allocation, nullptr);
    if (result != VK_SUCCESS)
        throw GpuError("VMA image allocation failed (VkResult " +
                       std::to_string(static_cast<int>(result)) + ")");

    m_image = vk::UniqueImage(rawImage, ctx.device());

    auto viewType = deduceViewType(desc.imageType, desc.arrayLayers);
    vk::ImageViewCreateInfo viewInfo(
        {}, *m_image, viewType, desc.format, {},
        {vk::ImageAspectFlagBits::eColor, 0, desc.mipLevels, 0, desc.arrayLayers});
    m_imageView = vkChecked([&] { return ctx.device().createImageViewUnique(viewInfo); },
                            "vkCreateImageView");
}

GpuImage::GpuImage(GpuContext &ctx, vk::Image image, vk::Format format, uint32_t width,
                   uint32_t height, vk::ImageViewType viewType, uint32_t mipLevels,
                   uint32_t arrayLayers, uint32_t depth)
    : m_ctx(&ctx), m_format(format), m_width(width), m_height(height), m_depth(depth),
      m_mipLevels(mipLevels), m_arrayLayers(arrayLayers), m_externalImage(true),
      m_layout(vk::ImageLayout::eUndefined)
{
    m_imageType = (viewType == vk::ImageViewType::e1D || viewType == vk::ImageViewType::e1DArray)
                      ? vk::ImageType::e1D
                  : (viewType == vk::ImageViewType::e3D) ? vk::ImageType::e3D
                                                         : vk::ImageType::e2D;

    m_image = vk::UniqueImage(image, ctx.device());
    vk::ImageViewCreateInfo viewInfo(
        {}, image, viewType, format, {},
        {vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, arrayLayers});
    m_imageView = vkChecked([&] { return ctx.device().createImageViewUnique(viewInfo); },
                            "vkCreateImageView");
}

GpuImage::~GpuImage()
{
    destroy();
}

GpuImage::GpuImage(GpuImage &&other) noexcept
    : m_ctx(nullptr), m_image(), m_imageView(), m_allocation(nullptr),
      m_format(vk::Format::eUndefined), m_width(0), m_height(0), m_depth(1), m_mipLevels(1),
      m_arrayLayers(1), m_imageType(vk::ImageType::e2D), m_externalImage(false),
      m_layout(vk::ImageLayout::eUndefined)
{
    swap(other);
}

GpuImage &GpuImage::operator=(GpuImage &&other) noexcept
{
    if (this != &other)
    {
        destroy();
        swap(other);
    }
    return *this;
}

vk::DescriptorImageInfo GpuImage::descriptorInfo() const
{
    return descriptorInfo(m_layout, {});
}

vk::DescriptorImageInfo GpuImage::descriptorInfo(vk::Sampler sampler) const
{
    return descriptorInfo(m_layout, sampler);
}

vk::DescriptorImageInfo GpuImage::descriptorInfo(vk::ImageLayout layout, vk::Sampler sampler) const
{
    return vk::DescriptorImageInfo(sampler, *m_imageView, layout);
}

size_t GpuImage::subresourceFootprint(uint32_t mipLevel) const
{
    const size_t width = mipExtent(m_width, mipLevel);
    const size_t height = mipExtent(m_height, mipLevel);
    const size_t depth = m_imageType == vk::ImageType::e3D ? mipExtent(m_depth, mipLevel) : 1;
    const size_t bytes = formatBytes(m_format);
    if (width > std::numeric_limits<size_t>::max() / height ||
        width * height > std::numeric_limits<size_t>::max() / depth ||
        width * height * depth > std::numeric_limits<size_t>::max() / bytes)
        throw GpuError("image subresource footprint overflows size_t");
    return width * height * depth * bytes;
}

void GpuImage::upload(const void *data, size_t size, vk::ImageLayout finalLayout, uint32_t mipLevel,
                      uint32_t arrayLayer)
{
    if (mipLevel >= m_mipLevels)
        throw GpuError("mipLevel out of range");
    if (arrayLayer >= m_arrayLayers)
        throw GpuError("arrayLayer out of range");
    if (size != subresourceFootprint(mipLevel))
        throw GpuError("image upload size does not match tightly-packed subresource footprint");

    detail::StagingScoped staging(*m_ctx, m_ctx->acquireStagingBuffer(size));
    std::memcpy(staging.get().mapped, data, size);

    GpuCommandBuffer cmd(*m_ctx);
    cmd.begin();
    cmd.imageBarrier(*m_image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                     vk::PipelineStageFlagBits::eTopOfPipe, {},
                     vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);

    vk::BufferImageCopy region(
        0, 0, 0,
        vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, mipLevel, arrayLayer, 1),
        vk::Offset3D(0, 0, 0),
        vk::Extent3D(mipExtent(m_width, mipLevel), mipExtent(m_height, mipLevel),
                     m_imageType == vk::ImageType::e3D ? mipExtent(m_depth, mipLevel) : 1));
    cmd.handle().copyBufferToImage(staging.get().buffer, *m_image,
                                   vk::ImageLayout::eTransferDstOptimal, 1, &region);

    vk::AccessFlags dstAccess;
    vk::PipelineStageFlags dstStage;
    if (finalLayout == vk::ImageLayout::eGeneral)
    {
        dstAccess = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        dstStage = vk::PipelineStageFlagBits::eComputeShader;
    }
    else if (finalLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        dstAccess = vk::AccessFlagBits::eShaderRead;
        dstStage = vk::PipelineStageFlagBits::eComputeShader;
    }
    else
    {
        dstAccess = vk::AccessFlagBits::eTransferRead;
        dstStage = vk::PipelineStageFlagBits::eTransfer;
    }

    cmd.imageBarrier(*m_image, vk::ImageLayout::eTransferDstOptimal, finalLayout,
                     vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite,
                     dstStage, dstAccess);
    cmd.submitAndWait();

    m_layout = finalLayout;
}

void GpuImage::download(void *data, size_t size, uint32_t mipLevel, uint32_t arrayLayer)
{
    if (mipLevel >= m_mipLevels)
        throw GpuError("mipLevel out of range");
    if (arrayLayer >= m_arrayLayers)
        throw GpuError("arrayLayer out of range");
    if (size != subresourceFootprint(mipLevel))
        throw GpuError("image download size does not match tightly-packed subresource footprint");

    detail::StagingScoped staging(*m_ctx, m_ctx->acquireStagingBuffer(size));
    GpuCommandBuffer cmd(*m_ctx);
    cmd.begin();
    cmd.imageBarrier(*m_image, m_layout, vk::ImageLayout::eTransferSrcOptimal,
                     vk::PipelineStageFlagBits::eAllCommands,
                     vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                     vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
    vk::BufferImageCopy region(
        0, 0, 0,
        vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, mipLevel, arrayLayer, 1),
        vk::Offset3D(0, 0, 0),
        vk::Extent3D(mipExtent(m_width, mipLevel), mipExtent(m_height, mipLevel),
                     m_imageType == vk::ImageType::e3D ? mipExtent(m_depth, mipLevel) : 1));
    cmd.handle().copyImageToBuffer(*m_image, vk::ImageLayout::eTransferSrcOptimal,
                                   staging.get().buffer, 1, &region);
    cmd.imageBarrier(*m_image, vk::ImageLayout::eTransferSrcOptimal, m_layout,
                     vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead,
                     vk::PipelineStageFlagBits::eAllCommands,
                     vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    cmd.submitAndWait();
    std::memcpy(data, staging.get().mapped, size);
}

void GpuImage::transition(GpuCommandBuffer &cmd, vk::ImageLayout newLayout,
                          vk::PipelineStageFlags srcStage, vk::AccessFlags srcAccess,
                          vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess)
{
    if (newLayout == m_layout)
        return;
    cmd.imageBarrier(*m_image, m_layout, newLayout, srcStage, srcAccess, dstStage, dstAccess);
    m_layout = newLayout;
}

void GpuImage::setDebugName(const std::string &name)
{
    if (!m_ctx)
        return;
    m_ctx->setDebugName(*m_image, name);
}

void GpuImage::destroy() noexcept
{
    if (!m_ctx)
        return;

    m_imageView.reset();

    if (m_externalImage)
    {
        m_image.release();
    }
    else if (m_allocation && m_ctx->allocator())
    {
        vmaDestroyImage(m_ctx->allocator(), static_cast<VkImage>(*m_image), m_allocation);
        m_allocation = nullptr;
        m_image.release();
    }
    else
    {
        m_image.reset();
    }
    m_ctx = nullptr;
}

void GpuImage::swap(GpuImage &other) noexcept
{
    std::swap(m_ctx, other.m_ctx);
    std::swap(m_image, other.m_image);
    std::swap(m_imageView, other.m_imageView);
    std::swap(m_allocation, other.m_allocation);
    std::swap(m_format, other.m_format);
    std::swap(m_width, other.m_width);
    std::swap(m_height, other.m_height);
    std::swap(m_depth, other.m_depth);
    std::swap(m_mipLevels, other.m_mipLevels);
    std::swap(m_arrayLayers, other.m_arrayLayers);
    std::swap(m_imageType, other.m_imageType);
    std::swap(m_layout, other.m_layout);
    std::swap(m_externalImage, other.m_externalImage);
}

vk::ImageViewType GpuImage::deduceViewType(vk::ImageType imageType, uint32_t arrayLayers) const
{
    switch (imageType)
    {
    case vk::ImageType::e1D:
        return (arrayLayers > 1) ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D;
    case vk::ImageType::e3D:
        return vk::ImageViewType::e3D;
    default:
        return (arrayLayers > 1) ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
    }
}

} // namespace pix
