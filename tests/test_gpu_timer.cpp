#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_timer.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pix;

TEST_CASE("GpuTimer pairCount", "[timer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuTimer timer(ctx, 3);
    REQUIRE(timer.pairCount() == 3);
}

TEST_CASE("GpuTimer begin/end readMs > 0", "[timer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuTimer timer(ctx, 1);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    timer.begin(cmd.handle());
    volatile float x = 0;
    for (int i = 0; i < 100000; ++i)
        x += 1.0f;
    (void)x;
    timer.end(cmd.handle());
    cmd.end();
    cmd.submitAndWait();

    double ms = timer.readMs(0);
    REQUIRE(ms >= 0.0);
}

TEST_CASE("GpuTimer multiple pairs", "[timer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuTimer timer(ctx, 2);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    timer.writeBegin(cmd.handle(), 0);
    volatile float x = 0;
    for (int i = 0; i < 50000; ++i)
        x += 1.0f;
    (void)x;
    timer.writeEnd(cmd.handle(), 0);

    timer.writeBegin(cmd.handle(), 1);
    for (int i = 0; i < 100000; ++i)
        x += 1.0f;
    timer.writeEnd(cmd.handle(), 1);
    cmd.end();
    cmd.submitAndWait();

    double ms0 = timer.readMs(0);
    double ms1 = timer.readMs(1);
    REQUIRE(ms0 >= 0.0);
    REQUIRE(ms1 >= 0.0);
}

TEST_CASE("GpuTimer resetAll then re-measure", "[timer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuTimer timer(ctx, 1);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    timer.resetAll(cmd.handle());
    timer.writeBegin(cmd.handle(), 0);
    volatile float x = 0;
    for (int i = 0; i < 10000; ++i)
        x += 1.0f;
    (void)x;
    timer.writeEnd(cmd.handle(), 0);
    cmd.end();
    cmd.submitAndWait();

    double ms = timer.readMs(0);
    REQUIRE(ms >= 0.0);
}
