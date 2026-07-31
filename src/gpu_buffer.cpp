#include "pixie_compute/gpu_buffer.hpp"

#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"

#include "error_utils.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>

namespace pix
{
using enum GpuBuffer::Type;

namespace
{
constexpr size_t kMaxStagingChunk = 64 * 1024 * 1024; // 64 MiB
}

// ---------------------------------------------------------------------------
// GpuBuffer
// ---------------------------------------------------------------------------

GpuBuffer::GpuBuffer(GpuContext &ctx, size_t size, Type type, vk::BufferUsageFlags extraUsage)
    : m_ctx(&ctx), m_size(size), m_type(type)
{
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferSrc |
                                 vk::BufferUsageFlagBits::eTransferDst;
    usage |= extraUsage;

    vk::BufferCreateInfo bufferInfo({}, size, usage);

    // Buffers are handed between the compute and transfer queue families
    // (e.g. upload via a transfer queue, then dispatched on the compute queue).
    // With the default eExclusive sharing mode that requires queue family
    // ownership transfers, which the library does not perform. Use concurrent
    // sharing when a dedicated transfer queue family exists instead.
    std::vector<uint32_t> sharingFamilies;
    if (ctx.m_hasSeparateTransferQueue)
    {
        sharingFamilies = {ctx.m_queueFamilyIndex, ctx.m_transferQueueFamilyIndex};
        bufferInfo.sharingMode = vk::SharingMode::eConcurrent;
        bufferInfo.setQueueFamilyIndices(sharingFamilies);
    }

    VmaAllocationCreateInfo allocInfo = {};
    if (type == Type::HostCoherent)
    {
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    else
    {
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }

    VmaAllocationInfo resultInfo = {};
    auto result =
        vmaCreateBuffer(ctx.allocator(), reinterpret_cast<const VkBufferCreateInfo *>(&bufferInfo),
                        &allocInfo, &m_buffer, &m_allocation, &resultInfo);
    if (result != VK_SUCCESS)
        throw GpuError("VMA buffer allocation failed (VkResult " +
                       std::to_string(static_cast<int>(result)) + ")");

    if (type == Type::HostCoherent)
        m_mapped = resultInfo.pMappedData;
}

GpuBuffer::~GpuBuffer()
{
    if (m_buffer && m_ctx)
    {
        vmaDestroyBuffer(m_ctx->allocator(), m_buffer, m_allocation);
    }
}

void GpuBuffer::swap(GpuBuffer &other) noexcept
{
    using std::swap;
    swap(m_ctx, other.m_ctx);
    swap(m_buffer, other.m_buffer);
    swap(m_allocation, other.m_allocation);
    swap(m_mapped, other.m_mapped);
    swap(m_size, other.m_size);
    swap(m_type, other.m_type);
}

GpuBuffer::GpuBuffer(GpuBuffer &&other) noexcept
    : m_ctx(other.m_ctx), m_buffer(other.m_buffer), m_allocation(other.m_allocation),
      m_mapped(other.m_mapped), m_size(other.m_size), m_type(other.m_type)
{
    other.m_ctx = nullptr;
    other.m_buffer = VK_NULL_HANDLE;
    other.m_allocation = nullptr;
    other.m_mapped = nullptr;
    other.m_size = 0;
    other.m_type = Type::Device;
}

GpuBuffer &GpuBuffer::operator=(GpuBuffer &&other) noexcept
{
    if (this != &other)
    {
        if (m_buffer && m_ctx)
        {
            vmaDestroyBuffer(m_ctx->allocator(), m_buffer, m_allocation);
            m_buffer = VK_NULL_HANDLE;
            m_allocation = nullptr;
        }
        swap(other);
    }
    return *this;
}

vk::DescriptorBufferInfo GpuBuffer::descriptorInfo(vk::DeviceSize offset,
                                                   vk::DeviceSize range) const
{
    return {m_buffer, offset, range};
}

// ---------------------------------------------------------------------------
// Synchronous upload / download (convenience wrappers)
// ---------------------------------------------------------------------------

void GpuBuffer::upload(const void *data, size_t size, size_t offset)
{
    if (!m_ctx)
        throw GpuError("upload on moved-from GpuBuffer");
    if (offset > m_size || size > m_size - offset)
        throw GpuError("upload exceeds buffer size");
    if (size == 0)
        return;

    if (m_type == Type::HostCoherent)
    {
        if (!m_mapped)
            throw GpuError("upload on unmapped buffer");
        std::memcpy(static_cast<char *>(m_mapped) + offset, data, size);
    }
    else
    {
        // Chunked staging: acquire per chunk, submit per chunk, release per chunk
        GpuCommandBuffer cmd(*m_ctx);
        size_t chunkOffset = 0;
        while (chunkOffset < size)
        {
            size_t chunkSize = std::min(size - chunkOffset, kMaxStagingChunk);
            auto staging = m_ctx->acquireStagingBuffer(chunkSize);
            std::memcpy(staging.mapped, static_cast<const char *>(data) + chunkOffset, chunkSize);

            cmd.reset();
            cmd.begin();
            cmd.copyBuffer(staging.buffer, m_buffer, chunkSize, 0, offset + chunkOffset);
            cmd.end();
            cmd.submitAndWait();

            m_ctx->releaseStagingBuffer(staging);
            chunkOffset += chunkSize;
        }
    }
}

void GpuBuffer::download(void *data, size_t size, size_t offset)
{
    if (!m_ctx)
        throw GpuError("download on moved-from GpuBuffer");
    if (offset > m_size || size > m_size - offset)
        throw GpuError("download exceeds buffer size");
    if (size == 0)
        return;

    if (m_type == Type::HostCoherent)
    {
        if (!m_mapped)
            throw GpuError("download on unmapped buffer");
        std::memcpy(data, static_cast<const char *>(m_mapped) + offset, size);
    }
    else
    {
        // Chunked staging: acquire per chunk, submit per chunk, memcpy, release
        GpuCommandBuffer cmd(*m_ctx);
        size_t chunkOffset = 0;
        while (chunkOffset < size)
        {
            size_t chunkSize = std::min(size - chunkOffset, kMaxStagingChunk);
            auto staging = m_ctx->acquireStagingBuffer(chunkSize);

            cmd.reset();
            cmd.begin();
            cmd.copyBuffer(m_buffer, staging.buffer, chunkSize, offset + chunkOffset, 0);
            cmd.end();
            cmd.submitAndWait();

            std::memcpy(static_cast<char *>(data) + chunkOffset, staging.mapped, chunkSize);
            m_ctx->releaseStagingBuffer(staging);
            chunkOffset += chunkSize;
        }
    }
}

void GpuBuffer::clear(uint32_t value)
{
    if (!m_ctx)
        throw GpuError("clear on moved-from GpuBuffer");

    if (m_type == Type::HostCoherent)
    {
        if (!m_mapped)
            throw GpuError("clear on unmapped buffer");
        if (value == 0)
        {
            std::memset(m_mapped, 0, m_size);
        }
        else
        {
            // Fill with 32-bit pattern
            size_t words = m_size / 4;
            auto *ptr = static_cast<uint32_t *>(m_mapped);
            for (size_t i = 0; i < words; ++i)
                ptr[i] = value;
            // Handle tail bytes (if size not multiple of 4)
            size_t tail = m_size % 4;
            if (tail > 0)
                std::memcpy(static_cast<char *>(m_mapped) + words * 4, &value, tail);
        }
    }
    else
    {
        // Device buffer: use GPU fillBuffer
        // vkCmdFillBuffer requires size to be multiple of 4 or VK_WHOLE_SIZE
        if (m_size < 4)
            return; // Buffer too small for 32-bit fill

        GpuCommandBuffer cmd(*m_ctx);
        cmd.begin();
        cmd.fillBuffer(m_buffer, value, VK_WHOLE_SIZE, 0);
        cmd.end();
        cmd.submitAndWait();
    }
}

void GpuBuffer::setDebugName(const std::string &name)
{
    if (!m_ctx)
        return;
    m_ctx->setDebugName(vk::Buffer(m_buffer), name);
}

// ---------------------------------------------------------------------------
// GpuBufferSlice forwarding
// ---------------------------------------------------------------------------

void GpuBufferSlice::upload(const void *data, size_t size, size_t uploadOffset)
{
    if (!m_buf)
        throw GpuError("upload on empty GpuBufferSlice");
    m_buf->upload(data, size, m_offset + uploadOffset);
}

void GpuBufferSlice::download(void *data, size_t size, size_t downloadOffset)
{
    if (!m_buf)
        throw GpuError("download on empty GpuBufferSlice");
    m_buf->download(data, size, m_offset + downloadOffset);
}

} // namespace pix
