#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_kernel.hpp"
#include "pixie_compute/gpu_stream.hpp"
#include "pixie_compute/utility.hpp"

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

using namespace pix;

// Shader: element-wise c = a + b
static constexpr const char *shader = R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<float> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c;

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

        // One kernel; resources are rebound by shader name per slot.
        GpuComputePipelineDesc desc;
        desc.slangSource = shader;
        desc.groupsX = numGroups;
        GpuKernel kernel(ctx, desc);

        GpuStream transfer(ctx, GpuCommandBuffer::QueueType::Transfer);
        GpuStream compute(ctx);

        // Prefetch: start uploading batch 0 into slot 0
        transfer.upload(*buf_a[0], std::span<const float>(h_a[0]));
        transfer.upload(*buf_b[0], std::span<const float>(h_b[0]));
        GpuEvent uploadDone = transfer.commit();

        //
        // Pipeline loop:
        //   Each iteration waits for the current batch's upload on the GPU,
        //   submits compute (non-blocking), then prefetches the next batch while
        //   compute runs.
        //
        //   Batch 0: [Upload]──[Compute]────────[Download]
        //   Batch 1:           [Upload]──[Compute]────────[Download]
        //   Batch 2:                     [Upload]──[Compute]────────[Download]
        //              ↑ overlap          ↑ overlap
        //
        float maxErr = 0.0f;
        GpuEvent computeDonePrev;

        for (uint32_t b = 0; b < numBatches; ++b)
        {
            int cur = b & 1;
            int nxt = (b + 1) & 1;

            // Rebind the kernel to the current slot and wait for its upload
            kernel.set("a", *buf_a[cur]);
            kernel.set("b", *buf_b[cur]);
            kernel.set("c", *buf_c[cur]);
            compute.wait(uploadDone);
            kernel.launch(compute, numGroups);
            GpuEvent computeDone = compute.commit();

            // Prefetch next batch — overlaps with compute. Wait only for the
            // compute that used this slot (two iterations ago), not the latest.
            if (b + 1 < numBatches)
            {
                if (computeDonePrev.valid())
                    transfer.wait(computeDonePrev);
                transfer.upload(*buf_a[nxt], std::span<const float>(h_a[b + 1]));
                transfer.upload(*buf_b[nxt], std::span<const float>(h_b[b + 1]));
                uploadDone = transfer.commit();
            }
            computeDonePrev = computeDone;

            // Wait for compute
            compute.wait();

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
