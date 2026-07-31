#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_image.hpp"
#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace pix;

// 16-bit single-channel (gray16) image read as unsigned int
static constexpr const char *shaderSource = R"(
RWStructuredBuffer<uint> output : register(u0);
Texture2D<uint> srcImage      : register(t1);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    srcImage.GetDimensions(w, h);
    if (tid.x < w && tid.y < h)
        output[tid.y * w + tid.x] = srcImage.Load(int3(tid.x, tid.y, 0));
}
)";

int main()
{
    try
    {
        GpuContext ctx;
        std::printf("[image_demo] GPU: %s\n", ctx.deviceName().c_str());

        auto spirv = compileSlangToSpirV(shaderSource).spirv;

        constexpr uint32_t W = 4, H = 4;
        constexpr uint32_t localX = 8, localY = 8;

        // ---- Gray16 input image ----
        // 16-bit unsigned single-channel (VK_FORMAT_R16_UINT, 2 bytes/pixel).
        // Requires shaderStorageImageExtendedFormats for eStorage usage.
        GpuImageDesc imageDesc;
        imageDesc.width = W;
        imageDesc.height = H;
        imageDesc.format = vk::Format::eR16Uint;
        imageDesc.usage = vk::ImageUsageFlagBits::eSampled |
                          vk::ImageUsageFlagBits::eTransferDst;
        GpuImage img(ctx, imageDesc);

        std::vector<uint16_t> pixels(W * H);
        for (uint32_t i = 0; i < W * H; ++i)
            pixels[i] = static_cast<uint16_t>(i * 4096);

        img.upload(pixels, vk::ImageLayout::eShaderReadOnlyOptimal);

        // ---- Output buffer (uint32_t — shader promotes uint16 to uint) ----
        GpuBuffer buf(ctx, W * H * sizeof(uint32_t), GpuBuffer::Type::HostCoherent);

        GpuComputePipelineDesc pipelineDesc;
        pipelineDesc.spirv = spirv;
        pipelineDesc.bindings = {
            GpuBinding(buf.descriptorInfo()),
            GpuBinding(img.descriptorInfo(), vk::DescriptorType::eSampledImage)};
        GpuComputePipeline pipeline(ctx, pipelineDesc);

        auto groups = workgroupCount2D(W, H, localX, localY);
        oneShotDispatch(ctx, pipeline, groups[0], groups[1], 1);

        std::vector<uint32_t> result(W * H, 0);
        buf.download(result);

        uint32_t maxErr = 0;
        for (uint32_t i = 0; i < W * H; ++i)
        {
            uint32_t expected = pixels[i];
            uint32_t actual = result[i];
            uint32_t err = (actual > expected) ? (actual - expected) : (expected - actual);
            if (err > maxErr)
                maxErr = err;
        }

        bool pass = maxErr == 0;
        std::printf("[image_demo] %ux%u R16  max error: %u  %s\n", W, H, maxErr,
                    pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
