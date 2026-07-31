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

static constexpr uint32_t WG_SIZE = 256;

static constexpr const char *reduce_shader = R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<float> input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> partial_sums;

groupshared float shared[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint gid = tid.x;
    shared[lid.x] = input[gid];
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128u; stride > 0u; stride >>= 1u)
    {
        if (lid.x < stride)
            shared[lid.x] += shared[lid.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid.x == 0u)
        partial_sums[tid.y] = shared[0];
}
)";

static constexpr const char *final_reduce_shader = R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<float> partial_sums;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> result;

groupshared float shared[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    shared[lid.x] = partial_sums[tid.x];
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128u; stride > 0u; stride >>= 1u)
    {
        if (lid.x < stride)
            shared[lid.x] += shared[lid.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid.x == 0u)
        result[0] = shared[0];
}
)";

int main()
{
    try
    {
        GpuContext ctx;
        std::printf("[reduce] GPU: %s\n", ctx.deviceName().c_str());

        constexpr uint32_t N = WG_SIZE * WG_SIZE;
        constexpr uint32_t numGroups = (N + WG_SIZE - 1) / WG_SIZE;

        std::vector<float> h_input(N);
        for (uint32_t i = 0; i < N; ++i)
            h_input[i] = 1.0f;

        double cpuSum = 0.0;
        for (auto v : h_input)
            cpuSum += v;

        GpuBuffer buf_input(ctx, N * sizeof(float), GpuBuffer::Type::HostCoherent);
        GpuBuffer buf_partial(ctx, numGroups * sizeof(float), GpuBuffer::Type::HostCoherent);
        GpuBuffer buf_result(ctx, sizeof(float), GpuBuffer::Type::HostCoherent);

        buf_input.upload(std::span<const float>(h_input));

        GpuComputePipelineDesc pass1Desc;
        pass1Desc.slangSource = reduce_shader;
        pass1Desc.bindings = {buf_input.descriptorInfo(), buf_partial.descriptorInfo()};
        pass1Desc.groupsX = WG_SIZE;
        GpuComputePipeline pass1(ctx, pass1Desc);

        GpuCommandBuffer cmd(ctx);
        cmd.begin();
        cmd.bind(pass1);
        cmd.dispatch(1, numGroups, 1);
        cmd.end();
        cmd.submitAndWait();

        GpuComputePipelineDesc pass2Desc;
        pass2Desc.slangSource = final_reduce_shader;
        pass2Desc.bindings = {buf_partial.descriptorInfo(), buf_result.descriptorInfo()};
        pass2Desc.groupsX = WG_SIZE;
        GpuComputePipeline pass2(ctx, pass2Desc);

        GpuCommandBuffer cmd2(ctx);
        cmd2.begin();
        cmd2.bind(pass2);
        cmd2.dispatch(1, 1, 1);
        cmd2.end();
        cmd2.submitAndWait();

        float gpuSum = 0.0f;
        buf_result.download(std::span<float>(&gpuSum, 1));

        float error = std::fabs(gpuSum - float(cpuSum));
        std::printf("[reduce] N=%u  groups=%u  cpu=%.1f  gpu=%.1f  error=%.6e  %s\n", N, numGroups,
                    cpuSum, double(gpuSum), error, error < 1.0f ? "PASS" : "FAIL");
        return error < 1.0f ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
