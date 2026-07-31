#include "pixie_compute/spirv_loader.hpp"

#include "pixie_compute/utility.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace pix
{

std::vector<uint32_t> loadSpirvFromMemory(const void *data, size_t size)
{
    if (size % 4 != 0)
        throw GpuError("SPIR-V size must be a multiple of 4 bytes");
    if (size < 4)
        throw GpuError("SPIR-V data is too small");

    const auto *bytes = static_cast<const uint8_t *>(data);
    uint32_t magic = 0;
    std::memcpy(&magic, bytes, 4);
    if (magic != kSpirvMagic)
        throw GpuError("not a valid SPIR-V module (bad magic number)");

    std::vector<uint32_t> words(size / 4);
    std::memcpy(words.data(), bytes, size);
    return words;
}

std::vector<uint32_t> loadSpirvFromFile(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw GpuError("failed to open SPIR-V file: " + path);

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
    if (!file.eof() && file.fail())
        throw GpuError("failed to read SPIR-V file: " + path);

    return loadSpirvFromMemory(bytes.data(), bytes.size());
}

} // namespace pix
