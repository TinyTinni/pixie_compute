#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"
#include "vector_add.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace pix;

TEST_CASE("compileSlangModule produces valid SPIR-V", "[shader][module]")
{
    auto result = compileSlangModule("vector_add", "main", {PIXIE_TEMPLATE_SHADER_DIR});
    REQUIRE_FALSE(result.spirv.empty());
    REQUIRE(result.spirv[0] == 0x07230203); // SPIR-V magic
}

TEST_CASE("vectorAdd rejects empty and mismatched inputs", "[gpu][vector_add]")
{
    auto &ctx = GpuTestFixture::ctx(); // skips the test if Vulkan is unavailable

    std::vector<float> a(16), b(8);
    REQUIRE_THROWS_AS(vectorAdd(ctx, a, b), GpuError);

    std::vector<float> empty;
    REQUIRE_THROWS_AS(vectorAdd(ctx, empty, empty), GpuError);
}

TEST_CASE("vectorAdd computes a + b on the GPU", "[gpu][vector_add]")
{
    auto &ctx = GpuTestFixture::ctx();

    constexpr uint32_t N = 1 << 16;
    std::vector<float> a(N), b(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) * 0.5f;
    }

    auto c = vectorAdd(ctx, a, b);

    REQUIRE(c.size() == N);
    float maxErr = 0.0f;
    for (uint32_t i = 0; i < N; ++i)
        maxErr = std::fmax(maxErr, std::fabs(c[i] - (a[i] + b[i])));
    REQUIRE(maxErr < 1e-4f);
}
