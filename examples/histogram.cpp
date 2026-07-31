#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <vector>

using namespace pix;

static constexpr const char *shader = R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> bins;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint val = input[tid.x];
    uint bin = val % 256u;
    InterlockedAdd(bins[bin], 1u);
}
)";

int main()
{
    try
    {
        GpuContext ctx;
        std::printf("[histogram] GPU: %s\n", ctx.deviceName().c_str());

        constexpr uint32_t N = 1 << 20;
        constexpr uint32_t wgSize = 256;
        constexpr uint32_t numBins = 256;

        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> dist(0, numBins - 1);
        std::vector<uint32_t> h_input(N);
        for (auto &v : h_input)
            v = dist(rng);

        std::vector<uint32_t> h_cpu_bins(numBins, 0);
        for (auto v : h_input)
            h_cpu_bins[v % numBins]++;

        GpuBuffer buf_input(ctx, N * sizeof(uint32_t), GpuBuffer::Type::HostCoherent);
        GpuBuffer buf_bins(ctx, numBins * sizeof(uint32_t), GpuBuffer::Type::HostCoherent);

        buf_input.upload(std::span<const uint32_t>(h_input));

        std::vector<uint32_t> zeros(numBins, 0);
        buf_bins.upload(std::span<const uint32_t>(zeros));

        GpuComputePipelineDesc desc;
        desc.slangSource = shader;
        desc.bindings = {buf_input.descriptorInfo(), buf_bins.descriptorInfo()};
        desc.groupsX = wgSize;
        GpuComputePipeline pipeline(ctx, desc);

        oneShotDispatch(ctx, pipeline, workgroupCount(N, wgSize), 1, 1);

        std::vector<uint32_t> h_gpu_bins(numBins);
        buf_bins.download(std::span<uint32_t>(h_gpu_bins));

        uint32_t mismatches = 0;
        uint32_t maxDiff = 0;
        for (uint32_t i = 0; i < numBins; ++i)
        {
            uint32_t diff = (h_gpu_bins[i] > h_cpu_bins[i]) ? h_gpu_bins[i] - h_cpu_bins[i]
                                                            : h_cpu_bins[i] - h_gpu_bins[i];
            if (diff > 0)
                ++mismatches;
            if (diff > maxDiff)
                maxDiff = diff;
        }

        bool pass = mismatches == 0;
        std::printf("[histogram] N=%u  bins=%u  mismatches=%u  max_diff=%u  %s\n", N, numBins,
                    mismatches, maxDiff, pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
