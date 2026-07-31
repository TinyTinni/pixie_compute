#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_profiler.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pix;

namespace
{

void requireTimestampSupport(const GpuContext &ctx)
{
    if (!ctx.limits().timestampComputeAndGraphics)
        SKIP("timestampComputeAndGraphics not supported");
}

} // namespace

TEST_CASE("GpuProfiler measures a frame section", "[profiler]")
{
    auto &ctx = GpuTestFixture::ctx();
    requireTimestampSupport(ctx);

    GpuProfiler profiler(ctx, 8);
    GpuBuffer buf(ctx, 4096, GpuBuffer::Type::Device);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    profiler.beginFrame(cmd.handle());
    {
        auto section = profiler.begin(cmd.handle(), "fill");
        cmd.fillBuffer(buf.handle(), 0x12345678u, 4096);
    }
    cmd.end();
    cmd.submitAndWait();

    auto sections = profiler.endFrame();
    REQUIRE(sections.size() == 1);
    REQUIRE(sections[0].name == "fill");
    REQUIRE(sections[0].ms >= 0.0);
}

TEST_CASE("GpuProfiler aggregates across frames", "[profiler]")
{
    auto &ctx = GpuTestFixture::ctx();
    requireTimestampSupport(ctx);

    GpuProfiler profiler(ctx, 8);
    GpuBuffer buf(ctx, 4096, GpuBuffer::Type::Device);

    for (int frame = 0; frame < 2; ++frame)
    {
        GpuCommandBuffer cmd(ctx);
        cmd.begin();
        profiler.beginFrame(cmd.handle());
        {
            auto section = profiler.begin(cmd.handle(), "fill");
            cmd.fillBuffer(buf.handle(), 0x12345678u, 4096);
        }
        cmd.end();
        cmd.submitAndWait();
        auto sections = profiler.endFrame();
        REQUIRE(sections.size() == 1);
    }

    auto &stats = profiler.stats().at("fill");
    REQUIRE(stats.frameCount == 2);
    REQUIRE(stats.lastMs >= 0.0);
    REQUIRE(stats.avgMs >= 0.0);
    REQUIRE(stats.minMs >= 0.0);
    REQUIRE(stats.maxMs >= 0.0);
    REQUIRE(stats.minMs <= stats.avgMs);
    REQUIRE(stats.avgMs <= stats.maxMs);
}

TEST_CASE("GpuProfiler exports aggregate timings", "[profiler]")
{
    auto &ctx = GpuTestFixture::ctx();
    requireTimestampSupport(ctx);
    GpuProfiler profiler(ctx, 4);
    GpuBuffer buf(ctx, 256, GpuBuffer::Type::Device);
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    profiler.beginFrame(cmd.handle());
    {
        auto section = profiler.begin(cmd.handle(), "exported");
        cmd.fillBuffer(buf.handle(), 1, 256);
    }
    cmd.end();
    cmd.submitAndWait();
    profiler.endFrame();

    REQUIRE(profiler.toJson().find("exported") != std::string::npos);
    REQUIRE(profiler.toCsv().find("name,last_ms") != std::string::npos);
    REQUIRE(profiler.toCsv().find("exported") != std::string::npos);
}

TEST_CASE("GpuProfiler nested sections", "[profiler]")
{
    auto &ctx = GpuTestFixture::ctx();
    requireTimestampSupport(ctx);

    GpuProfiler profiler(ctx, 8);
    GpuBuffer buf(ctx, 4096, GpuBuffer::Type::Device);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    profiler.beginFrame(cmd.handle());
    {
        auto outer = profiler.begin(cmd.handle(), "outer");
        {
            auto inner = profiler.begin(cmd.handle(), "inner");
            cmd.fillBuffer(buf.handle(), 0x1u, 4096);
        }
    }
    cmd.end();
    cmd.submitAndWait();

    auto sections = profiler.endFrame();
    REQUIRE(sections.size() == 2);
    REQUIRE(sections[0].name == "outer");
    REQUIRE(sections[1].name == "inner");
    REQUIRE(sections[0].ms >= 0.0);
    REQUIRE(sections[1].ms >= 0.0);
}

TEST_CASE("GpuProfiler beginSection before beginFrame throws", "[profiler]")
{
    auto &ctx = GpuTestFixture::ctx();
    requireTimestampSupport(ctx);

    GpuProfiler profiler(ctx, 4);
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    REQUIRE_THROWS_AS(profiler.beginSection(cmd.handle(), "x"), GpuError);
}

TEST_CASE("GpuProfiler unbalanced sections throw at endFrame", "[profiler]")
{
    auto &ctx = GpuTestFixture::ctx();
    requireTimestampSupport(ctx);

    GpuProfiler profiler(ctx, 4);
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    profiler.beginFrame(cmd.handle());
    profiler.beginSection(cmd.handle(), "open");
    cmd.end();
    cmd.submitAndWait();
    REQUIRE_THROWS_AS(profiler.endFrame(), GpuError);
}

TEST_CASE("GpuProfiler reset clears history", "[profiler]")
{
    auto &ctx = GpuTestFixture::ctx();
    requireTimestampSupport(ctx);

    GpuProfiler profiler(ctx, 4);
    GpuBuffer buf(ctx, 4096, GpuBuffer::Type::Device);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    profiler.beginFrame(cmd.handle());
    {
        auto section = profiler.begin(cmd.handle(), "fill");
        cmd.fillBuffer(buf.handle(), 0x2u, 4096);
    }
    cmd.end();
    cmd.submitAndWait();
    profiler.endFrame();

    REQUIRE(profiler.stats().size() == 1);
    profiler.reset();
    REQUIRE(profiler.stats().empty());
}
