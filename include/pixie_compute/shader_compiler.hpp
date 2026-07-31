#pragma once

#include "pixie_compute/detail/vulkan_include.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pix
{

/// Result of compiling a shader to SPIR-V.
struct CompiledShader
{
    std::vector<uint32_t> spirv;
    /// Size in bytes of the push-constant block declared in the shader (0 if none).
    uint32_t pushConstantSize = 0;
    /// Descriptor types of the shader's set-0 bindings, by binding index (reflected).
    std::vector<vk::DescriptorType> bindingTypes;
    /// Reflected set/binding names and descriptor types for Slang shaders.
    struct Binding
    {
        std::string name;
        uint32_t set = 0;
        uint32_t binding = 0;
        vk::DescriptorType type = vk::DescriptorType::eStorageBuffer;
    };
    std::vector<Binding> bindings;
};

/// Compile-time parameters for Slang shader compilation.
struct ShaderOptions
{
    /// Preprocessor defines applied to the source (name -> value), e.g.
    /// {"TILE_M", "16"}. Constants used in CoopMat template arguments, numthreads,
    /// etc. must be compile-time literals, so they can be supplied this way.
    std::vector<std::pair<std::string, std::string>> macros;
    /// Optional SPIR-V profile name, e.g. "spirv_1_6". The CooperativeMatrixKHR
    /// capability requires SPIR-V 1.6. Default (empty) is "cs_6_0", which
    /// emits SPIR-V 1.3, compatible with the library's Vulkan 1.1 instance.
    std::string spirvProfile;
};

/// Compile a Slang source string to SPIR-V. Results are cached by
/// source+entryPoint+searchPath+options.
CompiledShader compileSlangToSpirV(const std::string &source,
                                   const std::string &entryPoint = "main",
                                   const std::string &searchPath = "",
                                   const ShaderOptions &options = {});

/// Compile a named Slang module file to SPIR-V. Results are cached by
/// module name+entryPoint+searchPaths+options.
CompiledShader compileSlangModule(const std::string &moduleName,
                                  const std::string &entryPoint = "main",
                                  const std::vector<std::string> &searchPaths = {},
                                  const ShaderOptions &options = {});

/// Clear the internal shader compilation caches.
void clearShaderCache();

} // namespace pix
