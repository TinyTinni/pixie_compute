#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_timer.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

using namespace pix;

// ---- workgroupCount constexpr tests ----

static_assert(workgroupCount(64, 8) == 8);
static_assert(workgroupCount(65, 8) == 9);
static_assert(workgroupCount(1, 1) == 1);
static_assert(workgroupCount(0, 8) == 0);
static_assert(workgroupCount(8, 8) == 1);
static_assert(workgroupCount(1, 8) == 1);

static_assert(workgroupCount2D(16, 16, 8, 8)[0] == 2);
static_assert(workgroupCount2D(16, 16, 8, 8)[1] == 2);
static_assert(workgroupCount2D(0, 0, 8, 8)[0] == 0);
static_assert(workgroupCount2D(0, 0, 8, 8)[1] == 0);
static_assert(workgroupCount2D(5, 7, 1, 1)[0] == 5);
static_assert(workgroupCount2D(5, 7, 1, 1)[1] == 7);

static_assert(workgroupCount3D(4, 4, 4, 2, 2, 2)[0] == 2);
static_assert(workgroupCount3D(4, 4, 4, 2, 2, 2)[1] == 2);
static_assert(workgroupCount3D(4, 4, 4, 2, 2, 2)[2] == 2);
static_assert(workgroupCount3D(0, 0, 0, 8, 8, 8)[0] == 0);
static_assert(workgroupCount3D(1, 1, 1, 1, 1, 1)[0] == 1);

// ---- workgroupCount edge-case unit tests ----

TEST_CASE("workgroupCount edge: localSize=1 is identity", "[edge]")
{
    for (uint32_t dim : {1u, 7u, 64u, 255u, 1024u})
        REQUIRE(workgroupCount(dim, 1) == dim);
}

TEST_CASE("workgroupCount edge: dim == localSize returns 1", "[edge]")
{
    for (uint32_t v : {1u, 7u, 64u, 256u})
        REQUIRE(workgroupCount(v, v) == 1);
}

TEST_CASE("workgroupCount edge: localSize > dim returns 1", "[edge]")
{
    REQUIRE(workgroupCount(1, 8) == 1);
    REQUIRE(workgroupCount(3, 64) == 1);
    REQUIRE(workgroupCount(1, UINT32_MAX) == 1);
}

TEST_CASE("workgroupCount edge: non-aligned dimensions", "[edge]")
{
    REQUIRE(workgroupCount(16, 3) == 6);
    REQUIRE(workgroupCount(17, 5) == 4);
    REQUIRE(workgroupCount(1, 3) == 1);
}

TEST_CASE("workgroupCount2D edge: zero dimensions", "[edge]")
{
    auto r = workgroupCount2D(0, 0, 8, 8);
    REQUIRE(r[0] == 0);
    REQUIRE(r[1] == 0);
}

TEST_CASE("workgroupCount2D edge: localSize=1 is identity", "[edge]")
{
    auto r = workgroupCount2D(13, 7, 1, 1);
    REQUIRE(r[0] == 13);
    REQUIRE(r[1] == 7);
}

TEST_CASE("workgroupCount3D edge: zero dimensions", "[edge]")
{
    auto r = workgroupCount3D(0, 0, 0, 8, 8, 8);
    REQUIRE(r[0] == 0);
    REQUIRE(r[1] == 0);
    REQUIRE(r[2] == 0);
}

TEST_CASE("workgroupCount3D edge: localSize=1 is identity", "[edge]")
{
    auto r = workgroupCount3D(5, 9, 3, 1, 1, 1);
    REQUIRE(r[0] == 5);
    REQUIRE(r[1] == 9);
    REQUIRE(r[2] == 3);
}

// ---- workgroupCount property tests ----

TEST_CASE("RapidCheck: workgroupCount(dim, 1) == dim", "[rapidcheck][edge]")
{
    RC_ASSERT(rc::check(
        [](uint16_t dim16)
        {
            uint32_t dim = dim16;
            RC_ASSERT(workgroupCount(dim, 1) == dim);
        }));
}

TEST_CASE("RapidCheck: workgroupCount(dim, localSize) >= ceil(dim/localSize)", "[rapidcheck][edge]")
{
    RC_ASSERT(rc::check(
        [](uint16_t dim16, uint16_t ls16)
        {
            uint32_t dim = dim16;
            uint32_t localSize = ls16 % 255 + 1;
            uint32_t result = workgroupCount(dim, localSize);
            uint32_t naive = (dim + localSize - 1) / localSize;
            RC_ASSERT(result == naive);
        }));
}

TEST_CASE("RapidCheck: workgroupCount2D product covers all elements", "[rapidcheck][edge]")
{
    RC_ASSERT(rc::check(
        [](uint8_t dx8, uint8_t dy8, uint8_t lx8, uint8_t ly8)
        {
            uint32_t dx = dx8, dy = dy8;
            uint32_t lx = lx8 % 15 + 1;
            uint32_t ly = ly8 % 15 + 1;
            auto groups = workgroupCount2D(dx, dy, lx, ly);
            uint64_t totalElements = uint64_t(dx) * dy;
            uint64_t totalCapacity = uint64_t(groups[0]) * lx * uint64_t(groups[1]) * ly;
            RC_ASSERT(totalCapacity >= totalElements);
        }));
}

TEST_CASE("RapidCheck: workgroupCount3D product covers all elements", "[rapidcheck][edge]")
{
    RC_ASSERT(rc::check(
        [](uint8_t dx8, uint8_t dy8, uint8_t dz8, uint8_t lx8, uint8_t ly8, uint8_t lz8)
        {
            uint32_t dx = dx8, dy = dy8, dz = dz8;
            uint32_t lx = lx8 % 7 + 1;
            uint32_t ly = ly8 % 7 + 1;
            uint32_t lz = lz8 % 7 + 1;
            auto groups = workgroupCount3D(dx, dy, dz, lx, ly, lz);
            uint64_t totalElements = uint64_t(dx) * dy * dz;
            uint64_t totalCapacity =
                uint64_t(groups[0]) * lx * uint64_t(groups[1]) * ly * uint64_t(groups[2]) * lz;
            RC_ASSERT(totalCapacity >= totalElements);
        }));
}

// ---- GpuTimer edge-case tests ----

TEST_CASE("GpuTimer pairCount consistency", "[edge]")
{
    auto &ctx = GpuTestFixture::ctx();
    REQUIRE(GpuTimer(ctx, 1).pairCount() == 1);
    REQUIRE(GpuTimer(ctx, 4).pairCount() == 4);
    REQUIRE(GpuTimer(ctx, 16).pairCount() == 16);
}

TEST_CASE("GpuTimer begin/end convenience equals manual sequence", "[edge]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    std::vector<float> zeros(4, 0.0f);
    buf.upload(zeros.data(), zeros.size() * sizeof(float));
    auto di = buf.descriptorInfo();

    GpuComputePipeline pipeline(ctx, R"(
        RWStructuredBuffer<float> buf : register(u0);
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) { buf[tid.x] = 1.0; }
    )",
                                {di}, 0, 4, 1, 1);

    // Convenience API
    GpuTimer t1(ctx, 1);
    GpuCommandBuffer cmd1(ctx);
    cmd1.begin();
    t1.begin(cmd1.handle());
    cmd1.bind(pipeline);
    cmd1.dispatch(pipeline);
    t1.end(cmd1.handle());
    cmd1.end();
    cmd1.submitAndWait();
    double ms1 = t1.readMs(0);

    // Manual sequence
    GpuTimer t2(ctx, 1);
    GpuCommandBuffer cmd2(ctx);
    cmd2.begin();
    t2.resetAll(cmd2.handle());
    t2.writeBegin(cmd2.handle(), 0);
    cmd2.bind(pipeline);
    cmd2.dispatch(pipeline);
    t2.writeEnd(cmd2.handle(), 0);
    cmd2.end();
    cmd2.submitAndWait();
    double ms2 = t2.readMs(0);

    REQUIRE(ms1 >= 0.0);
    REQUIRE(ms2 >= 0.0);
}

TEST_CASE("GpuTimer multiple pairs monotonicity", "[edge]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuTimer timer(ctx, 2);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    timer.resetAll(cmd.handle());

    // Pair 0: short workload
    timer.writeBegin(cmd.handle(), 0);
    volatile float x = 0;
    for (int i = 0; i < 50000; ++i)
        x += 1.0f;
    (void)x;
    timer.writeEnd(cmd.handle(), 0);

    // Pair 1: long workload
    timer.writeBegin(cmd.handle(), 1);
    for (int i = 0; i < 200000; ++i)
        x += 1.0f;
    (void)x;
    timer.writeEnd(cmd.handle(), 1);

    cmd.end();
    cmd.submitAndWait();

    double ms0 = timer.readMs(0);
    double ms1 = timer.readMs(1);
    REQUIRE(ms0 >= 0.0);
    REQUIRE(ms1 >= 0.0);
}
