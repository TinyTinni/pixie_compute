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

// Shader: element-wise c = a + b
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
        GpuContext ctx;
        std::printf("[pipeline] GPU: %s\n", ctx.deviceName().c_str());
        if (ctx.hasSeparateTransferQueue())
            std::printf("[pipeline] separate transfer queue available\n");

        constexpr uint32_t wgSize = 256;
        constexpr uint32_t batchSize = 16384;
        constexpr uint32_t numBatches = 8;
        constexpr uint32_t numGroups = batchSize / wgSize;

        // Prepare all host data up front
        std::vector<std::vector<float>> h_a(numBatches, std::vector<float>(batchSize));
        std::vector<std::vector<float>> h_b(numBatches, std::vector<float>(batchSize));
        std::vector<float> h_c(batchSize);

        for (uint32_t b = 0; b < numBatches; ++b)
            for (uint32_t i = 0; i < batchSize; ++i)
            {
                h_a[b][i] = static_cast<float>(b * batchSize + i);
                h_b[b][i] = static_cast<float>(b * batchSize + i) * 0.5f;
            }

        // Double-buffered device slots (ping-pong)
        auto buf_a0 = GpuBuffer(ctx, batchSize * sizeof(float), GpuBuffer::Type::Device);
        auto buf_b0 = GpuBuffer(ctx, batchSize * sizeof(float), GpuBuffer::Type::Device);
        auto buf_c0 = GpuBuffer(ctx, batchSize * sizeof(float), GpuBuffer::Type::Device);
        auto buf_a1 = GpuBuffer(ctx, batchSize * sizeof(float), GpuBuffer::Type::Device);
        auto buf_b1 = GpuBuffer(ctx, batchSize * sizeof(float), GpuBuffer::Type::Device);
        auto buf_c1 = GpuBuffer(ctx, batchSize * sizeof(float), GpuBuffer::Type::Device);
        GpuBuffer *buf_a[2] = {&buf_a0, &buf_a1};
        GpuBuffer *buf_b[2] = {&buf_b0, &buf_b1};
        GpuBuffer *buf_c[2] = {&buf_c0, &buf_c1};

        GpuComputePipeline pipes[2] = {
            GpuComputePipeline(ctx, shader,
                               {
                                   buf_a0.descriptorInfo(),
                                   buf_b0.descriptorInfo(),
                                   buf_c0.descriptorInfo(),
                               },
                               0, wgSize, 1, 1),
            GpuComputePipeline(ctx, shader,
                               {
                                   buf_a1.descriptorInfo(),
                                   buf_b1.descriptorInfo(),
                                   buf_c1.descriptorInfo(),
                               },
                               0, wgSize, 1, 1),
        };

        // Prefetch: start uploading batch 0 into slot 0
        GpuCommandBuffer uploadCmd(ctx, GpuCommandBuffer::QueueType::Transfer);
        uploadCmd.begin();
        uploadCmd.upload(*buf_a[0], std::span<const float>(h_a[0]));
        uploadCmd.upload(*buf_b[0], std::span<const float>(h_b[0]));
        uploadCmd.submit();

        //
        // Pipeline loop:
        //   Each iteration waits for the current batch's upload, submits compute
        //   (non-blocking), then prefetches the next batch while compute runs.
        //
        //   Batch 0: [Upload]──[Compute]────────[Download]
        //   Batch 1:           [Upload]──[Compute]────────[Download]
        //   Batch 2:                     [Upload]──[Compute]────────[Download]
        //              ↑ overlap          ↑ overlap
        //
        float maxErr = 0.0f;

        for (uint32_t b = 0; b < numBatches; ++b)
        {
            int cur = b & 1;
            int nxt = (b + 1) & 1;

            // Wait for current batch upload
            uploadCmd.wait();

            // Submit compute (non-blocking)
            GpuCommandBuffer computeCmd(ctx);
            computeCmd.begin();
            computeCmd.bind(pipes[cur]);
            computeCmd.dispatch(pipes[cur], numGroups, 1, 1);
            computeCmd.submit();

            // Prefetch next batch — overlaps with compute
            if (b + 1 < numBatches)
            {
                uploadCmd = GpuCommandBuffer(ctx, GpuCommandBuffer::QueueType::Transfer);
                uploadCmd.begin();
                uploadCmd.upload(*buf_a[nxt], std::span<const float>(h_a[b + 1]));
                uploadCmd.upload(*buf_b[nxt], std::span<const float>(h_b[b + 1]));
                uploadCmd.submit();
            }

            // Wait for compute
            computeCmd.wait();

            // Download and verify
            buf_c[cur]->download(std::span<float>(h_c));

            for (uint32_t i = 0; i < batchSize; ++i)
            {
                float ref = h_a[b][i] + h_b[b][i];
                maxErr = std::fmax(maxErr, std::fabs(h_c[i] - ref));
            }
        }

        std::printf("[pipeline] batches=%u  batchSize=%u  max error: %.6e  %s\n", numBatches,
                    batchSize, maxErr, maxErr < 1e-6f ? "PASS" : "FAIL");
        return maxErr < 1e-6f ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
