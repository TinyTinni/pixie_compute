#include "embedded_shaders_shaders.hpp"
#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/spirv_loader.hpp"
#include "pixie_compute/utility.hpp"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <span>
#include <vector>

using namespace pix;

int main()
{
    try
    {
        GpuContext ctx;
        std::printf("[embedded_shaders] GPU: %s\n", ctx.deviceName().c_str());

        // The shader below is compiled to SPIR-V at build time by slangc and
        // embedded via cmake/PixieShaders.cmake — no runtime shader compiler.
        constexpr uint32_t N = 4096;
        constexpr uint32_t wgSize = 256;

        std::vector<float> h_a(N), h_b(N), h_ref(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            h_a[i] = static_cast<float>(i);
            h_b[i] = static_cast<float>(i) * 0.5f;
            h_ref[i] = h_a[i] + h_b[i];
        }

        GpuBuffer buf_a(ctx, N * sizeof(float), GpuBuffer::Type::HostCoherent);
        GpuBuffer buf_b(ctx, N * sizeof(float), GpuBuffer::Type::HostCoherent);
        GpuBuffer buf_c(ctx, N * sizeof(float), GpuBuffer::Type::HostCoherent);

        buf_a.upload(std::span<const float>(h_a));
        buf_b.upload(std::span<const float>(h_b));

        auto spirv =
            loadSpirvFromMemory(shaders::vector_add, shaders::vector_add_count * sizeof(uint32_t));

        GpuComputePipeline pipeline(
            ctx, spirv, {buf_a.descriptorInfo(), buf_b.descriptorInfo(), buf_c.descriptorInfo()}, 0,
            wgSize, 1, 1);

        oneShotDispatch(ctx, pipeline, workgroupCount(N, wgSize), 1, 1);

        std::vector<float> h_c(N);
        buf_c.download(std::span<float>(h_c));

        float maxErr = 0.0f;
        for (uint32_t i = 0; i < N; ++i)
            maxErr = std::fmax(maxErr, std::fabs(h_c[i] - h_ref[i]));

        std::printf("[embedded_shaders] N=%u  max error: %.6e  %s\n", N, maxErr,
                    maxErr < 1e-6f ? "PASS" : "FAIL");
        return maxErr < 1e-6f ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
