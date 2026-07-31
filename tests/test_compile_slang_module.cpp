#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace pix;

static std::string createTempSlangFile(const char *filename, const char *source)
{
    auto tmpDir = std::filesystem::temp_directory_path();
    auto path = tmpDir / filename;

    std::ofstream f(path);
    f << source;
    f.close();
    return path.string();
}

TEST_CASE("compileSlangModule produces valid SPIR-V", "[shader][module]")
{
    auto path = createTempSlangFile("test_module.slang", R"(
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )");

    auto dir = std::filesystem::path(path).parent_path().string();
    auto spirv = compileSlangModule("test_module", "main", {dir}).spirv;
    REQUIRE_FALSE(spirv.empty());
    REQUIRE(spirv[0] == 0x07230203); // SPIR-V magic

    std::filesystem::remove(path);
}

TEST_CASE("compileSlangModule reflects push constant size", "[shader][module]")
{
    auto path = createTempSlangFile("pc_module.slang", R"(
        [[vk::push_constant]]
        cbuffer PushConstants { float scale; };
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )");

    auto dir = std::filesystem::path(path).parent_path().string();
    auto result = compileSlangModule("pc_module", "main", {dir});
    REQUIRE(result.pushConstantSize == sizeof(float));

    std::filesystem::remove(path);
}

TEST_CASE("compileSlangModule caches results", "[shader][module]")
{
    auto path = createTempSlangFile("cached_module.slang", R"(
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )");

    auto dir = std::filesystem::path(path).parent_path().string();
    auto a = compileSlangModule("cached_module", "main", {dir}).spirv;
    auto b = compileSlangModule("cached_module", "main", {dir}).spirv;
    REQUIRE(a == b);

    std::filesystem::remove(path);
}

TEST_CASE("compileSlangModule fails on bad entry point", "[shader][module]")
{
    auto path = createTempSlangFile("bad_entry.slang", R"(
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {}
    )");

    auto dir = std::filesystem::path(path).parent_path().string();
    REQUIRE_THROWS_AS(compileSlangModule("bad_entry", "nonexistent", {dir}), GpuError);

    std::filesystem::remove(path);
}

TEST_CASE("compileSlangModule fails on missing module", "[shader][module]")
{
    REQUIRE_THROWS_AS(compileSlangModule("nonexistent_module", "main", {}), GpuError);
}
