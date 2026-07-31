#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

using namespace pix;

TEST_CASE("GpuError carries message", "[error]")
{
    GpuError err("boom");
    REQUIRE(std::string(err.what()) == "boom");
}

TEST_CASE("workgroupCount correctness", "[utility]")
{
    REQUIRE(workgroupCount(64, 8) == 8);
    REQUIRE(workgroupCount(65, 8) == 9);
    REQUIRE(workgroupCount(1, 8) == 1);
    REQUIRE(workgroupCount(0, 8) == 0);
    REQUIRE(workgroupCount(7, 7) == 1);
    REQUIRE(workgroupCount(8, 7) == 2);
}

TEST_CASE("workgroupCount with default localSize", "[utility]")
{
    REQUIRE(workgroupCount(64) == 8);
    REQUIRE(workgroupCount(65) == 9);
}

TEST_CASE("compileSlangToSpirV produces valid SPIR-V", "[shader]")
{
    static const char *src = R"(
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )";
    auto spirv = compileSlangToSpirV(src, "main").spirv;
    REQUIRE_FALSE(spirv.empty());
    REQUIRE(spirv[0] == 0x07230203); // SPIR-V magic
}

TEST_CASE("compileSlangToSpirV caches results", "[shader]")
{
    static const char *src = R"(
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )";
    auto a = compileSlangToSpirV(src, "main").spirv;
    auto b = compileSlangToSpirV(src, "main").spirv;
    REQUIRE(a == b);
}

TEST_CASE("compileSlangToSpirV reflects push constant size", "[shader][reflection]")
{
    REQUIRE(compileSlangToSpirV(R"(
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )",
                                "main")
                .pushConstantSize == 0);

    REQUIRE(compileSlangToSpirV(R"(
        cbuffer Globals : register(b0) { float scale; };
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )",
                                "main")
                .pushConstantSize == 0);

    REQUIRE(compileSlangToSpirV(R"(
        [[vk::push_constant]]
        cbuffer PushConstants : register(b0) { float scale; };
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )",
                                "main")
                .pushConstantSize == 4);

    REQUIRE(compileSlangToSpirV(R"(
        [[vk::push_constant]]
        cbuffer PushConstants : register(b0) { float a; float b; };
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )",
                                "main")
                .pushConstantSize == 8);

    REQUIRE(compileSlangToSpirV(R"(
        [[vk::push_constant]]
        cbuffer PushConstants : register(b0) { float4 v; };
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )",
                                "main")
                .pushConstantSize == 16);

    REQUIRE(compileSlangToSpirV(R"(
        struct Params { float a; float b; float c; float d; };
        [[vk::push_constant]] ConstantBuffer<Params> pc;
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )",
                                "main")
                .pushConstantSize == 16);
}

TEST_CASE("compileSlangToSpirV fails on bad entry point", "[shader]")
{
    static const char *src = R"(
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )";
    REQUIRE_THROWS_AS(compileSlangToSpirV(src, "nonexistent"), GpuError);
}

// A cooperative-matrix kernel whose tile shape and thread count are controlled
// by compile-time macros. Mirrors the coop kernel used by the example.
static const char *kCoopMatKernel = R"(
    using namespace linalg;

    #ifndef TILE_M
    #define TILE_M 16
    #endif
    #ifndef SUBGROUP_SIZE
    #define SUBGROUP_SIZE 32
    #endif

    RWStructuredBuffer<int8_t>  a : register(u0);
    RWStructuredBuffer<int8_t>  b : register(u1);
    RWStructuredBuffer<int32_t> c : register(u2);

    [numthreads(SUBGROUP_SIZE, 1, 1)]
    void main(uint3 gid : SV_GroupID)
    {
        CoopMat<int32_t, MemoryScope.Subgroup, TILE_M, TILE_M, CoopMatMatrixUse::MatrixAccumulator> acc;
        acc.fill(int32_t(0));

        let aTile = coopMatLoad<int8_t, MemoryScope.Subgroup, TILE_M, TILE_M, CoopMatMatrixUse::MatrixA, CoopMatMatrixLayout::RowMajor>(a, 0, TILE_M);
        let bTile = coopMatLoad<int8_t, MemoryScope.Subgroup, TILE_M, TILE_M, CoopMatMatrixUse::MatrixB, CoopMatMatrixLayout::RowMajor>(b, 0, TILE_M);

        acc = coopMatMulAdd<int32_t, false>(aTile, bTile, acc);
        acc.Store<CoopMatMatrixLayout::RowMajor>(c, 0, TILE_M);
    }
)";

TEST_CASE("compileSlangToSpirV targets SPIR-V 1.6 when requested", "[shader][coopmat]")
{
    ShaderOptions options;
    options.spirvProfile = "spirv_1_6";
    auto spirv = compileSlangToSpirV(kCoopMatKernel, "main", "", options).spirv;
    REQUIRE_FALSE(spirv.empty());
    REQUIRE(spirv[0] == 0x07230203);      // SPIR-V magic
    REQUIRE(spirv[1] == 0x00010600);      // version word: 1.6
}

TEST_CASE("compileSlangToSpirV defaults to SPIR-V 1.3", "[shader]")
{
    static const char *src = R"(
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )";
    auto spirv = compileSlangToSpirV(src, "main").spirv;
    REQUIRE_FALSE(spirv.empty());
    REQUIRE(spirv[0] == 0x07230203);      // SPIR-V magic
    REQUIRE(spirv[1] == 0x00010300);      // version word: 1.3
}

TEST_CASE("compileSlangToSpirV macros reach the shader", "[shader][coopmat]")
{
    ShaderOptions options;
    options.spirvProfile = "spirv_1_6";
    options.macros = {{"TILE_M", "8"}};
    auto eight = compileSlangToSpirV(kCoopMatKernel, "main", "", options).spirv;
    REQUIRE_FALSE(eight.empty());

    options.macros = {{"TILE_M", "16"}};
    auto sixteen = compileSlangToSpirV(kCoopMatKernel, "main", "", options).spirv;
    REQUIRE_FALSE(sixteen.empty());

    REQUIRE(eight != sixteen);            // tile shape reached the template args

    auto sixteenCached = compileSlangToSpirV(kCoopMatKernel, "main", "", options).spirv;
    REQUIRE(sixteen == sixteenCached);    // same options hit the cache
}

TEST_CASE("RapidCheck: workgroupCount matches naive formula", "[rapidcheck]")
{
    auto props = rc::check(
        [](uint16_t dim16, uint16_t ls16)
        {
            uint32_t dim = dim16;
            uint32_t localSize = static_cast<uint16_t>(ls16 % 255 + 1); // 1..255
            auto result = workgroupCount(dim, localSize);
            auto expected = (dim + localSize - 1) / localSize;
            RC_ASSERT(result == expected);
        });
    REQUIRE(props);
}

TEST_CASE("workgroupCount2D correctness", "[utility]")
{
    auto result = workgroupCount2D(17, 33, 8, 8);
    REQUIRE(result[0] == 3); // ceil(17/8)
    REQUIRE(result[1] == 5); // ceil(33/8)
}

TEST_CASE("workgroupCount3D correctness", "[utility]")
{
    auto result = workgroupCount3D(17, 33, 5, 8, 8, 4);
    REQUIRE(result[0] == 3); // ceil(17/8)
    REQUIRE(result[1] == 5); // ceil(33/8)
    REQUIRE(result[2] == 2); // ceil(5/4)
}

TEST_CASE("GpuContext device properties", "[context]")
{
    auto &ctx = GpuTestFixture::ctx();
    REQUIRE_FALSE(ctx.deviceName().empty());
    REQUIRE(ctx.limits().maxComputeWorkGroupCount[0] > 0);
    REQUIRE(ctx.timestampPeriod() > 0.0f);
}

TEST_CASE("GpuContext pipeline cache save/load", "[context]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto saved = ctx.savePipelineCache();
    REQUIRE_FALSE(saved.empty());

    ctx.loadPipelineCache(saved);
    auto saved2 = ctx.savePipelineCache();
    REQUIRE_FALSE(saved2.empty());
}
