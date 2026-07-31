#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <queue>
#include <span>
#include <thread>
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

// ---------------------------------------------------------------------------
// Thread-safe work queue
// ---------------------------------------------------------------------------

class WorkQueue
{
    public:
    void push(uint32_t batch)
    {
        std::lock_guard lock(m_mutex);
        m_queue.push(batch);
    }

    bool pop(uint32_t &batch)
    {
        std::lock_guard lock(m_mutex);
        if (m_queue.empty())
            return false;
        batch = m_queue.front();
        m_queue.pop();
        return true;
    }

    private:
    std::queue<uint32_t> m_queue;
    std::mutex m_mutex;
};

// ---------------------------------------------------------------------------
// Per-thread resources (created on main thread to avoid Slang races)
// ---------------------------------------------------------------------------

struct WorkerState
{
    GpuBuffer buf_a;
    GpuBuffer buf_b;
    GpuBuffer buf_c;
    GpuComputePipeline pipe;

    WorkerState(GpuContext &ctx, uint32_t batchSize)
        : buf_a(ctx, batchSize * sizeof(float), GpuBuffer::Type::Device),
          buf_b(ctx, batchSize * sizeof(float), GpuBuffer::Type::Device),
          buf_c(ctx, batchSize * sizeof(float), GpuBuffer::Type::Device),
          pipe(ctx, shader,
               {
                   buf_a.descriptorInfo(),
                   buf_b.descriptorInfo(),
                   buf_c.descriptorInfo(),
               },
               0, 256, 1, 1)
    {
    }
};

// ---------------------------------------------------------------------------
// Worker: takes ownership of a WorkerState, processes batches from the queue
// ---------------------------------------------------------------------------

void worker(GpuContext &ctx, WorkQueue &queue, WorkerState state,
            const std::vector<std::vector<float>> &h_a, const std::vector<std::vector<float>> &h_b,
            uint32_t batchSize, uint32_t numGroups, std::atomic<uint32_t> &errorCount)
{
    std::vector<float> h_c(batchSize);
    uint32_t batch;

    while (queue.pop(batch))
    {
        // Upload
        GpuCommandBuffer upCmd(ctx, GpuCommandBuffer::QueueType::Transfer);
        upCmd.begin();
        upCmd.upload(state.buf_a, std::span<const float>(h_a[batch]));
        upCmd.upload(state.buf_b, std::span<const float>(h_b[batch]));
        upCmd.submitAndWait();

        // Compute
        GpuCommandBuffer cmd(ctx);
        cmd.begin();
        cmd.bind(state.pipe);
        cmd.dispatch(state.pipe, numGroups, 1, 1);
        cmd.submitAndWait();

        // Download and verify
        state.buf_c.download(std::span<float>(h_c));

        for (uint32_t i = 0; i < batchSize; ++i)
        {
            float ref = h_a[batch][i] + h_b[batch][i];
            if (std::fabs(h_c[i] - ref) > 1e-6f)
                errorCount.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main()
{
    try
    {
        GpuContext ctx;
        std::printf("[pipeline_mt] GPU: %s\n", ctx.deviceName().c_str());
        if (ctx.hasSeparateTransferQueue())
            std::printf("[pipeline_mt] separate transfer queue available\n");

        constexpr uint32_t batchSize = 16384;
        constexpr uint32_t numBatches = 32;
        constexpr uint32_t numGroups = batchSize / 256;

        unsigned int hwThreads = std::thread::hardware_concurrency();
        if (hwThreads == 0)
            hwThreads = 4;
        uint32_t numThreads = hwThreads > 8 ? 8 : hwThreads;

        std::printf("[pipeline_mt] batches=%u  batchSize=%u  threads=%u\n", numBatches, batchSize,
                    numThreads);

        // Prepare all host data up front
        std::vector<std::vector<float>> h_a(numBatches, std::vector<float>(batchSize));
        std::vector<std::vector<float>> h_b(numBatches, std::vector<float>(batchSize));

        for (uint32_t b = 0; b < numBatches; ++b)
            for (uint32_t i = 0; i < batchSize; ++i)
            {
                h_a[b][i] = static_cast<float>(b * batchSize + i);
                h_b[b][i] = static_cast<float>(b * batchSize + i) * 0.5f;
            }

        // Fill work queue
        WorkQueue queue;
        for (uint32_t b = 0; b < numBatches; ++b)
            queue.push(b);

        // Pre-create per-thread resources on the main thread (Slang is not
        // thread-safe for concurrent pipeline creation).
        std::vector<WorkerState> states;
        states.reserve(numThreads);
        for (uint32_t t = 0; t < numThreads; ++t)
            states.emplace_back(ctx, batchSize);

        // Launch workers
        std::atomic<uint32_t> errorCount{0};
        std::vector<std::thread> threads;

        for (uint32_t t = 0; t < numThreads; ++t)
            threads.emplace_back(worker, std::ref(ctx), std::ref(queue), std::move(states[t]),
                                 std::ref(h_a), std::ref(h_b), batchSize, numGroups,
                                 std::ref(errorCount));

        for (auto &t : threads)
            t.join();

        std::printf("[pipeline_mt] errors: %u  %s\n", errorCount.load(),
                    errorCount == 0 ? "PASS" : "FAIL");
        return errorCount == 0 ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
