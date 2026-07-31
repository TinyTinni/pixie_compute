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

struct PushConstants
{
    uint32_t samples_per_thread;
    uint32_t pad[3];
};

static constexpr const char *shader = R"(
RWStructuredBuffer<uint> hit_counts : register(u0);

struct Params {
    uint samples_per_thread;
    uint pad0;
    uint pad1;
    uint pad2;
};
[[vk::push_constant]] cbuffer PC : register(b0) { Params params; };

uint xorshift(uint state)
{
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

float uniform01(inout uint state)
{
    state = xorshift(state);
    return float(state) / 4294967295.0;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    uint rng = idx * 1664525u + 1013904223u;
    if (rng == 0u) rng = 1u;

    uint hits = 0u;
    for (uint i = 0u; i < params.samples_per_thread; ++i)
    {
        float x = uniform01(rng) * 2.0 - 1.0;
        float y = uniform01(rng) * 2.0 - 1.0;
        if (x * x + y * y <= 1.0)
            ++hits;
    }
    hit_counts[idx] = hits;
}
)";

int main()
{
    try
    {
        GpuContext ctx;
        std::printf("[monte_carlo_pi] GPU: %s\n", ctx.deviceName().c_str());

        constexpr uint32_t threadsPerGroup = 256;
        constexpr uint32_t numGroups = 64;
        constexpr uint32_t totalThreads = threadsPerGroup * numGroups;
        constexpr uint32_t samplesPerThread = 4096;
        constexpr uint64_t totalSamples = uint64_t(totalThreads) * samplesPerThread;

        GpuBuffer buf_hits(ctx, totalThreads * sizeof(uint32_t), GpuBuffer::Type::HostCoherent);

        PushConstants pc{};
        pc.samples_per_thread = samplesPerThread;

        GpuComputePipeline pipeline(ctx, shader,
                                    {
                                        buf_hits.descriptorInfo(),
                                    },
                                    sizeof(pc), threadsPerGroup, 1, 1);

        GpuCommandBuffer cmd(ctx);
        cmd.begin();
        cmd.bind(pipeline);
        cmd.pushConstants(pipeline, &pc, sizeof(pc));
        cmd.dispatch(pipeline, numGroups, 1, 1);
        cmd.end();
        cmd.submitAndWait();

        std::vector<uint32_t> h_hits(totalThreads);
        buf_hits.download(std::span<uint32_t>(h_hits));

        uint64_t totalHits = 0;
        for (auto h : h_hits)
            totalHits += h;

        double pi_est = 4.0 * double(totalHits) / double(totalSamples);
        double error = std::fabs(pi_est - M_PI);

        std::printf("[monte_carlo_pi] samples=%llu  pi=%.8f  error=%.6e  %s\n",
                    (unsigned long long)totalSamples, pi_est, error,
                    error < 0.01 ? "PASS" : "FAIL");
        return error < 0.01 ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
