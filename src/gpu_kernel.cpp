#include "pixie_compute/gpu_kernel.hpp"

#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_image.hpp"
#include "pixie_compute/gpu_stream.hpp"
#include "pixie_compute/utility.hpp"

#include <cstring>

namespace pix
{

GpuKernel::GpuKernel(GpuContext &ctx, const GpuComputePipelineDesc &desc) : m_pipeline(ctx, desc)
{
}

const CompiledShader::Binding *GpuKernel::findBinding(std::string_view name) const noexcept
{
    for (const auto &info : m_pipeline.bindingInfo())
        if (info.set == 0 && info.name == name)
            return &info;
    return nullptr;
}

void GpuKernel::set(std::string_view name, const GpuBinding &binding)
{
    m_pipeline.updateBinding(name, binding);
}

void GpuKernel::set(std::string_view name, GpuBuffer &buffer, vk::DeviceSize offset,
                    vk::DeviceSize range)
{
    const CompiledShader::Binding *info = findBinding(name);
    if (!info)
        throw GpuError("shader binding not found: " + std::string(name));
    m_pipeline.updateBinding(name, GpuBinding(buffer.descriptorInfo(offset, range), info->type));
}

void GpuKernel::set(std::string_view name, GpuImage &image)
{
    const CompiledShader::Binding *info = findBinding(name);
    if (!info)
        throw GpuError("shader binding not found: " + std::string(name));
    const vk::ImageLayout layout = info->type == vk::DescriptorType::eSampledImage
                                       ? vk::ImageLayout::eShaderReadOnlyOptimal
                                       : vk::ImageLayout::eGeneral;
    m_pipeline.updateBinding(name, GpuBinding(image.descriptorInfo(layout), info->type));
}

void GpuKernel::set(std::string_view name, GpuImage &image, vk::ImageLayout layout)
{
    const CompiledShader::Binding *info = findBinding(name);
    if (!info)
        throw GpuError("shader binding not found: " + std::string(name));
    m_pipeline.updateBinding(name, GpuBinding(image.descriptorInfo(layout), info->type));
}

void GpuKernel::setPushConstants(const void *data, size_t size)
{
    const size_t expected = m_pipeline.pushConstantSize();
    if (expected == 0)
        throw GpuError("kernel has no push-constant block declared by the shader");
    if (size != expected)
        throw GpuError("kernel push constants must be exactly " + std::to_string(expected) +
                       " bytes (shader-declared block size), got " + std::to_string(size));
    m_pushConstants.resize(size);
    if (size != 0)
        std::memcpy(m_pushConstants.data(), data, size);
}

void GpuKernel::validateReady() const
{
    if (!m_pipeline.bindingsComplete())
        throw GpuError("kernel launch attempted before all named bindings were set");
    if (m_pipeline.pushConstantSize() != 0 &&
        m_pushConstants.size() != m_pipeline.pushConstantSize())
        throw GpuError("kernel launch requires a complete push-constant block");
}

void GpuKernel::bindAndPush(GpuCommandBuffer &command)
{
    validateReady();
    command.bind(m_pipeline);
    if (!m_pushConstants.empty())
        command.pushConstants(m_pushConstants.data(), m_pushConstants.size());
}

void GpuKernel::bindAndPush(GpuStream &stream)
{
    validateReady();
    stream.bind(m_pipeline);
    if (!m_pushConstants.empty())
        stream.pushConstants(m_pushConstants.data(), m_pushConstants.size());
}

void GpuKernel::launch(GpuCommandBuffer &command, uint32_t groupsX, uint32_t groupsY,
                       uint32_t groupsZ)
{
    bindAndPush(command);
    command.dispatch(groupsX, groupsY, groupsZ);
}

void GpuKernel::launch(GpuCommandBuffer &command)
{
    bindAndPush(command);
    command.dispatch();
}

void GpuKernel::launch(GpuStream &stream, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
{
    bindAndPush(stream);
    stream.dispatch(groupsX, groupsY, groupsZ);
}

void GpuKernel::launch(GpuStream &stream)
{
    bindAndPush(stream);
    stream.dispatch();
}

} // namespace pix
