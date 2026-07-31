#include "pixie_compute/gpu_context.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace pix;

TEST_CASE("infoDump contains device and environment summary", "[info_dump]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto dump = ctx.infoDump();

    REQUIRE_FALSE(dump.empty());
    REQUIRE(dump.find("pixie_compute info dump") != std::string::npos);
    REQUIRE(dump.find(ctx.deviceName()) != std::string::npos);
    REQUIRE(dump.find("queueFamilies") != std::string::npos);
    REQUIRE(dump.find("instanceExtensions") != std::string::npos);
    REQUIRE(dump.find("deviceExtensions") != std::string::npos);
    REQUIRE(dump.find("memoryHeaps") != std::string::npos);
    REQUIRE(dump.find("memoryStats") != std::string::npos);
    REQUIRE(dump.find("totalUsage") != std::string::npos);
}

TEST_CASE("infoDump lists enabled debug-utils extension when available", "[info_dump]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto dump = ctx.infoDump();
    if (ctx.debugUtilsEnabled())
        REQUIRE(dump.find("VK_EXT_debug_utils") != std::string::npos);
}
