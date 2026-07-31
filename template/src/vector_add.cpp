#include "vector_add.hpp"

#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"

#ifndef PIXIE_TEMPLATE_SHADER_DIR
#error "PIXIE_TEMPLATE_SHADER_DIR must be defined by the build (see CMakeLists.txt)"
#endif

namespace pix
{

std::vector<float> vectorAdd(GpuContext &ctx, std::span<const float> a, std::span<const float> b)
{
    if (a.size() != b.size())
        throw GpuError("vectorAdd: input sizes differ");
    if (a.empty())
        throw GpuError("vectorAdd: empty input");

    constexpr uint32_t workgroupSize = 256;
    const uint32_t n = static_cast<uint32_t>(a.size());

    // HostCoherent = CPU-visible GPU memory: upload()/download() are plain
    // memcpy. For large write-once data prefer GpuBuffer::Type::Device (staged).
    GpuBuffer bufA(ctx, a.size_bytes(), GpuBuffer::Type::HostCoherent);
    GpuBuffer bufB(ctx, b.size_bytes(), GpuBuffer::Type::HostCoherent);
    GpuBuffer bufC(ctx, a.size_bytes(), GpuBuffer::Type::HostCoherent);

    bufA.upload(a);
    bufB.upload(b);

    // Compile shaders/vector_add.slang to SPIR-V (cached internally by the lib).
    auto compiled = compileSlangModule("vector_add", "main", {PIXIE_TEMPLATE_SHADER_DIR});

    // The descriptorInfo() order must match the register(uN) slots in the shader.
    GpuComputePipelineDesc pipelineDesc;
    pipelineDesc.spirv = compiled.spirv;
    pipelineDesc.pushConstantSize = compiled.pushConstantSize;
    pipelineDesc.bindings = {
        bufA.descriptorInfo(), bufB.descriptorInfo(), bufC.descriptorInfo()};
    GpuComputePipeline pipeline(ctx, pipelineDesc);

    // One-shot: begin/bind/dispatch/submit+wait. workgroupCount() rounds up.
    oneShotDispatch(ctx, pipeline, workgroupCount(n, workgroupSize), 1, 1);

    std::vector<float> out(a.size());
    bufC.download(out);
    return out;
}

} // namespace pix
