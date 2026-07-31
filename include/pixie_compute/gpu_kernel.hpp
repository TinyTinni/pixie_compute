#pragma once

#include "pixie_compute/gpu_compute_pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pix
{

class GpuImage;
class GpuStream;
class GpuBuffer;

/// Higher-level compute kernel wrapper with named bindings and launch helpers.
class GpuKernel
{
    public:
    GpuKernel(GpuContext &ctx, const GpuComputePipelineDesc &desc);

    GpuKernel(const GpuKernel &) = delete;
    GpuKernel &operator=(const GpuKernel &) = delete;
    GpuKernel(GpuKernel &&) noexcept = default;
    GpuKernel &operator=(GpuKernel &&) noexcept = default;

    /// Bind a resource by its reflected Slang name. Names are taken as
    /// std::string_view, so string literals are passed without allocation.
    void set(std::string_view name, const GpuBinding &binding);
    void set(std::string_view name, GpuBuffer &buffer, vk::DeviceSize offset = 0,
             vk::DeviceSize range = VK_WHOLE_SIZE);
    void set(std::string_view name, GpuImage &image);
    void set(std::string_view name, GpuImage &image, vk::ImageLayout layout);

    /// Set the full push-constant block. The size must exactly match the shader's
    /// declared push-constant block size, otherwise a GpuError is thrown.
    void setPushConstants(const void *data, size_t size);
    template <typename T> void setPushConstants(const T &value)
    {
        setPushConstants(&value, sizeof(T));
    }

    void launch(GpuCommandBuffer &command, uint32_t groupsX, uint32_t groupsY = 1,
                uint32_t groupsZ = 1);
    void launch(GpuCommandBuffer &command);
    void launch(GpuStream &stream, uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1);
    void launch(GpuStream &stream);

    GpuComputePipeline &pipeline() noexcept { return m_pipeline; }
    const GpuComputePipeline &pipeline() const noexcept { return m_pipeline; }

    private:
    void validateReady() const;
    void bindAndPush(GpuCommandBuffer &command);
    void bindAndPush(GpuStream &stream);

    const CompiledShader::Binding *findBinding(std::string_view name) const noexcept;

    GpuComputePipeline m_pipeline;
    std::vector<std::byte> m_pushConstants;
};

} // namespace pix
