#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_timeline_semaphore.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pix;

TEST_CASE("setDebugName and labels record without tripping VVL", "[debug_utils]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuBuffer buf(ctx, 1024, GpuBuffer::Type::Device);
    buf.setDebugName("test_buffer");

    GpuTimelineSemaphore sem = ctx.createTimelineSemaphore();
    sem.setDebugName("test_sem");

    GpuCommandBuffer cmd(ctx);
    cmd.setDebugName("test_cmd");
    cmd.begin();
    cmd.beginLabel("outer", {1.0f, 0.0f, 0.0f, 1.0f});
    {
        GpuCommandBuffer::ScopedLabel inner(cmd, "inner");
        cmd.fillBuffer(buf.handle(), 0u, 1024);
    }
    cmd.endLabel();
    cmd.end();
    cmd.submitAndWait();
    SUCCEED();
}

TEST_CASE("GpuContext::setDebugName on raw handles is safe", "[debug_utils]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuBuffer buf(ctx, 4096, GpuBuffer::Type::HostCoherent);
    GpuTimelineSemaphore sem = ctx.createTimelineSemaphore();

    REQUIRE_NOTHROW(ctx.setDebugName(buf.handle(), "raw_buffer"));
    REQUIRE_NOTHROW(ctx.setDebugName(sem.handle(), "raw_sem"));
    REQUIRE_NOTHROW(ctx.debugUtilsEnabled());
}

TEST_CASE("breakOnError context runs normally without validation errors", "[debug_utils]")
{
    GpuContextDesc desc;
    desc.enableValidation = true;
    desc.breakOnError = true;
    GpuContext ctx(desc);
    if (!ctx.validationEnabled())
        SKIP("validation layer not available");

    GpuBuffer buf(ctx, 4096, GpuBuffer::Type::Device);
    uint32_t value = 42;
    buf.upload(&value, sizeof(value));
    uint32_t out = 0;
    buf.download(&out, sizeof(out));
    REQUIRE(out == 42);
}
