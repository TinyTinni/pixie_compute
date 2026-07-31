#include "pixie_compute/spirv_loader.hpp"
#include "pixie_compute/utility.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{

std::vector<uint32_t> sampleWords()
{
    return {0x07230203u, 0x00010000u, 0x00000000u, 0x00000000u};
}

} // namespace

TEST_CASE("loadSpirvFromMemory accepts a valid header", "[spirv]")
{
    auto words = sampleWords();
    auto spirv = pix::loadSpirvFromMemory(words.data(), words.size() * sizeof(uint32_t));
    REQUIRE(spirv == words);
}

TEST_CASE("loadSpirvFromMemory rejects non-word-aligned size", "[spirv]")
{
    auto words = sampleWords();
    REQUIRE_THROWS_AS(pix::loadSpirvFromMemory(words.data(), 3), pix::GpuError);
    REQUIRE_THROWS_AS(pix::loadSpirvFromMemory(words.data(), 5), pix::GpuError);
}

TEST_CASE("loadSpirvFromMemory rejects data too small", "[spirv]")
{
    uint8_t byte = 0x07;
    REQUIRE_THROWS_AS(pix::loadSpirvFromMemory(&byte, 1), pix::GpuError);
}

TEST_CASE("loadSpirvFromMemory rejects bad magic", "[spirv]")
{
    std::vector<uint32_t> words = {0xdeadbeefu, 0x00000000u};
    REQUIRE_THROWS_AS(pix::loadSpirvFromMemory(words.data(), words.size() * sizeof(uint32_t)),
                      pix::GpuError);
}

TEST_CASE("loadSpirvFromFile round-trips a temp file", "[spirv]")
{
    auto path = std::filesystem::temp_directory_path() / "pixie_spirv_roundtrip.bin";
    {
        auto words = sampleWords();
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(words.data()),
                  static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    }
    auto spirv = pix::loadSpirvFromFile(path.string());
    REQUIRE(spirv == sampleWords());
    std::filesystem::remove(path);
}

TEST_CASE("loadSpirvFromFile throws on missing file", "[spirv]")
{
    auto path = std::filesystem::temp_directory_path() / "pixie_spirv_does_not_exist.bin";
    REQUIRE_THROWS_AS(pix::loadSpirvFromFile(path.string()), pix::GpuError);
}
