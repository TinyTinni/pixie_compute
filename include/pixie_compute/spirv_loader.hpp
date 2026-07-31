#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pix
{

/// SPIR-V magic number identifying a valid module (0x07230203).
constexpr uint32_t kSpirvMagic = 0x07230203;

/// Load a SPIR-V binary file into a vector of 32-bit words.
/// Throws GpuError on I/O failure, non-word-aligned size, or invalid magic.
std::vector<uint32_t> loadSpirvFromFile(const std::string &path);

/// Interpret raw bytes as SPIR-V. Throws GpuError if size is not a multiple of 4
/// or the magic number does not match.
std::vector<uint32_t> loadSpirvFromMemory(const void *data, size_t size);

} // namespace pix
