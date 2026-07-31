#include "pixie_compute/renderdoc_capture.hpp"

#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <span>
#include <vector>

using namespace pix;

static constexpr const char *shader = R"(
RWStructuredBuffer<float> a : register(u0);
RWStructuredBuffer<float> b : register(u1);
RWStructuredBuffer<float> c : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    c[tid.x] = a[tid.x] + b[tid.x];
}
)";

int main()
{
    try
    {
        // Construct the capture object before the GpuContext so RenderDoc (if present)
        // can hook instance/device creation.
        RenderDocCapture capture;

        if (capture.available())
        {
            std::printf("[renderdoc_capture] RenderDoc API found");
            if (capture.targetControlConnected())
                std::printf(", connected to the RenderDoc UI");
            std::printf("\n");
        }
        else
        {
            std::printf("[renderdoc_capture] RenderDoc not available; running without capture\n");
        }

        GpuContext ctx;
        std::printf("[renderdoc_capture] GPU: %s\n", ctx.deviceName().c_str());

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

        GpuComputePipeline pipeline(ctx, shader,
                                    {
                                        buf_a.descriptorInfo(),
                                        buf_b.descriptorInfo(),
                                        buf_c.descriptorInfo(),
                                    },
                                    0, wgSize, 1, 1);

        // Record the dispatch inside a capture region. For headless compute there is
        // no presentation frame, so a capture is explicitly delimited this way.
        if (capture.available())
        {
            ScopedCapture scoped(capture);
            oneShotDispatch(ctx, pipeline, workgroupCount(N, wgSize), 1, 1);
            std::printf("[renderdoc_capture] capture triggered\n");
        }
        else
        {
            oneShotDispatch(ctx, pipeline, workgroupCount(N, wgSize), 1, 1);
        }

        std::vector<float> h_c(N);
        buf_c.download(std::span<float>(h_c));

        float maxErr = 0.0f;
        for (uint32_t i = 0; i < N; ++i)
            maxErr = std::fmax(maxErr, std::fabs(h_c[i] - h_ref[i]));

        std::printf("[renderdoc_capture] N=%u  max error: %.6e  %s\n", N, maxErr,
                    maxErr < 1e-6f ? "PASS" : "FAIL");
        return maxErr < 1e-6f ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
