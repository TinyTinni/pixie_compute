#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_timeline_semaphore.hpp"
#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

using namespace pix;

TEST_CASE("GpuCommandBuffer begin/end lifecycle", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuCommandBuffer cmd(ctx);

    REQUIRE_FALSE(cmd.recording());
    cmd.begin();
    REQUIRE(cmd.recording());
    cmd.end();
    REQUIRE_FALSE(cmd.recording());
}

TEST_CASE("GpuCommandBuffer reset", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuCommandBuffer cmd(ctx);

    cmd.begin();
    cmd.reset();
    REQUIRE_FALSE(cmd.recording());

    cmd.begin();
    REQUIRE(cmd.recording());
    cmd.end();
}

TEST_CASE("GpuCommandBuffer fillBuffer", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, sizeof(uint32_t) * 4, GpuBuffer::Type::HostCoherent);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.fillBuffer(buf.handle(), 0xDEADBEEF, buf.size());
    cmd.submitAndWait();

    std::vector<uint32_t> result(4, 0);
    buf.download(std::span<uint32_t>(result));
    for (auto v : result)
        REQUIRE(v == 0xDEADBEEF);
}

TEST_CASE("GpuCommandBuffer copyBuffer", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> srcData = {10, 20, 30, 40};
    GpuBuffer src(ctx, srcData.size() * sizeof(uint32_t), GpuBuffer::Type::HostCoherent);
    src.upload(std::span<const uint32_t>(srcData));

    GpuBuffer dst(ctx, srcData.size() * sizeof(uint32_t), GpuBuffer::Type::Device);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.copyBuffer(src.handle(), dst.handle(), src.size());
    cmd.submitAndWait();

    std::vector<uint32_t> result(4, 0);
    cmd.begin();
    cmd.download(dst, std::span<uint32_t>(result));
    cmd.submitAndWait();
    REQUIRE(result == srcData);
}

TEST_CASE("GpuCommandBuffer bufferBarrier then copy", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> srcData = {1, 2, 3};
    GpuBuffer src(ctx, srcData.size() * sizeof(uint32_t), GpuBuffer::Type::HostCoherent);
    src.upload(std::span<const uint32_t>(srcData));

    GpuBuffer dst(ctx, srcData.size() * sizeof(uint32_t), GpuBuffer::Type::HostCoherent);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bufferBarrier(src.handle(), 0, src.size(), vk::PipelineStageFlagBits::eTransfer,
                      vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eTransfer,
                      vk::AccessFlagBits::eTransferRead);
    cmd.copyBuffer(src.handle(), dst.handle(), src.size());
    cmd.bufferBarrier(dst.handle(), 0, dst.size(), vk::PipelineStageFlagBits::eTransfer,
                      vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eHost,
                      vk::AccessFlagBits::eHostRead);
    cmd.submitAndWait();

    std::vector<uint32_t> result(3, 0);
    dst.download(std::span<uint32_t>(result));
    REQUIRE(result == srcData);
}

TEST_CASE("GpuCommandBuffer submitAndWait is idempotent", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.end();
    cmd.submitAndWait();
}

TEST_CASE("GpuCommandBuffer move semantics", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuCommandBuffer cmd(ctx);
    cmd.begin();

    GpuCommandBuffer moved = std::move(cmd);
    REQUIRE(moved.recording());
    moved.end();
    REQUIRE_FALSE(moved.recording());
}

TEST_CASE("GpuCommandBuffer non-blocking submit and wait", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.end();
    cmd.submit();
    cmd.wait();
}

TEST_CASE("oneShotDispatch convenience", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compileSlangToSpirV(R"(
        RWStructuredBuffer<float> buffer : register(u0);
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID)
        {
            buffer[tid.x] = 42.0;
        }
    )")
                     .spirv;

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    std::vector<float> zeros(4, 0.0f);
    buf.upload(std::span<const float>(zeros));

    auto di = buf.descriptorInfo();
    GpuComputePipeline pipeline(ctx, spirv, {di}, 0, 4, 1, 1);

    oneShotDispatch(ctx, pipeline);

    std::vector<float> result(4, 0.0f);
    buf.download(std::span<float>(result));
    for (auto v : result)
        REQUIRE(v == 42.0f);
}

TEST_CASE("GpuCommandBuffer timeline semaphore signal", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuBuffer buf(ctx, sizeof(uint32_t) * 4, GpuBuffer::Type::HostCoherent);
    std::vector<uint32_t> zeros(4, 0);
    buf.upload(std::span<const uint32_t>(zeros));

    GpuTimelineSemaphore sem(ctx);
    uint64_t val = sem.next();

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.fillBuffer(buf.handle(), 0x42, buf.size());
    cmd.end();
    cmd.submit({}, 0, sem, val);
    cmd.wait();

    std::vector<uint32_t> result(4, 0);
    buf.download(std::span<uint32_t>(result));
    for (auto v : result)
        REQUIRE(v == 0x42);
}

TEST_CASE("semaphore wait gates transfer commands in waiting batch", "[cmd]")
{
    auto &ctx = GpuTestFixture::ctx();

    auto spirv = compileSlangToSpirV(R"(
        RWStructuredBuffer<float> buffer : register(u0);
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID)
        {
            buffer[tid.x] = 42.0;
        }
    )")
                     .spirv;

    GpuBuffer src(ctx, sizeof(float) * 4, GpuBuffer::Type::Device);
    GpuBuffer dst(ctx, sizeof(float) * 4, GpuBuffer::Type::Device);

    auto di = src.descriptorInfo();
    GpuComputePipeline pipeline(ctx, spirv, {di}, 0, 4, 1, 1);

    GpuTimelineSemaphore sem(ctx);
    uint64_t val = sem.next();

    // The waiting submission only records a transfer command (copyBuffer).
    // The timeline wait must gate transfer-stage work, otherwise the copy can
    // read src while the dispatch is still writing it.
    GpuCommandBuffer computeCmd(ctx);
    computeCmd.begin();
    computeCmd.bind(pipeline);
    computeCmd.dispatch(pipeline);
    computeCmd.submit({}, 0, sem, val);

    GpuCommandBuffer transferCmd(ctx);
    transferCmd.begin();
    transferCmd.copyBuffer(src.handle(), dst.handle(), src.size());
    transferCmd.submitAndWait(sem, val);

    std::vector<float> result(4, 0.0f);
    dst.download(std::span<float>(result));
    for (auto v : result)
        REQUIRE(v == 42.0f);
}
