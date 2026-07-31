#pragma once

#include "pixie_compute/detail/vulkan_include.hpp"
#include "pixie_compute/shader_compiler.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace pix
{

class GpuContext;
class GpuCommandBuffer;

/// One shader binding slot: a buffer or image resource plus its descriptor type.
///
/// For Slang shaders, the descriptor type is reflected from the shader and the
/// type field is validated against the reflection result. For SPIR-V shaders,
/// the type must be specified explicitly for image bindings.
struct GpuBinding
{
    std::variant<vk::DescriptorBufferInfo, vk::DescriptorImageInfo> resource;
    vk::DescriptorType type;

    /// Buffer binding; storage buffer by default.
    GpuBinding(const vk::DescriptorBufferInfo &buf,
               vk::DescriptorType type = vk::DescriptorType::eStorageBuffer)
        : resource(buf), type(type)
    {
    }

    /// Image binding. For Slang shaders the descriptor type is reflected and this
    /// parameter may be omitted; for SPIR-V shaders pass the exact type.
    GpuBinding(const vk::DescriptorImageInfo &img,
               vk::DescriptorType type = vk::DescriptorType::eSampledImage)
        : resource(img), type(type)
    {
    }
};

/// Configuration for GpuComputePipeline construction.
struct GpuComputePipelineDesc
{
    /// SPIR-V words. Mutually exclusive with slangSource.
    std::vector<uint32_t> spirv;
    /// Slang source string (requires the slang build option). When set, descriptor
    /// types and the push-constant size are reflected from the shader.
    std::string slangSource;
    /// Set-0 bindings, ordered by binding index.
    std::vector<GpuBinding> bindings;
    /// Push constant block size in bytes; 0 = reflect (Slang) or none (SPIR-V).
    uint32_t pushConstantSize = 0;
    /// Default dispatch group counts used by GpuCommandBuffer::dispatch() with no counts.
    uint32_t groupsX = 1, groupsY = 1, groupsZ = 1;
    uint32_t maxDescriptorSets = 16;
    std::string entryPoint = "main";
};

/// Vulkan compute pipeline with automatic descriptor set management.
class GpuComputePipeline
{
    friend class GpuCommandBuffer;

    public:
    /// Create a pipeline from a GpuComputePipelineDesc config.
    GpuComputePipeline(GpuContext &ctx, const GpuComputePipelineDesc &desc);

    ~GpuComputePipeline();

    GpuComputePipeline(const GpuComputePipeline &) = delete;
    GpuComputePipeline &operator=(const GpuComputePipeline &) = delete;
    GpuComputePipeline(GpuComputePipeline &&) noexcept;
    GpuComputePipeline &operator=(GpuComputePipeline &&) noexcept;

    /// Update descriptor bindings for set 0.
    void updateBindings(const std::vector<GpuBinding> &bindings);
    /// Update one set-zero binding by its reflected Slang name. Accepts a string
    /// literal, std::string, or std::string_view without copying the name.
    void updateBinding(std::string_view name, const GpuBinding &binding);
    /// Update the descriptor bindings of a specific binding set instance.
    ///
    /// Note: the pipeline layout always describes a single descriptor set (set 0),
    /// so the "binding sets" managed here are interchangeable instances of that
    /// set — useful for ping-pong/double-buffering, not for shaders that declare
    /// multiple descriptor sets (set > 0 is rejected by Slang reflection).
    void updateBindingSet(uint32_t bindingSetIndex, const std::vector<GpuBinding> &bindings);
    /// Allocate and bind a new descriptor set instance, returning its index.
    uint32_t addBindingSet(const std::vector<GpuBinding> &bindings);
    /// Pre-allocate descriptor set instances without binding data.
    void allocateBindingSets(uint32_t count);
    /// Clear all descriptor set instances. Invalidates any previously returned
    /// indices from addBindingSet.
    void clearBindingSets();

    /// Assign a debug name to the pipeline (VK_EXT_debug_utils). No-op when the
    /// extension is unavailable.
    void setDebugName(const std::string &name);

    vk::Pipeline handle() const noexcept { return *m_pipeline; }
    vk::PipelineLayout layout() const noexcept { return *m_layout; }
    /// Number of allocated descriptor set instances (see addBindingSet).
    uint32_t bindingSetCount() const noexcept
    {
        return static_cast<uint32_t>(m_descriptorSets.size());
    }
    /// Size in bytes of the push-constant range in the pipeline layout.
    uint32_t pushConstantSize() const noexcept { return m_pushConstantSize; }
    /// Returns {groupsX, groupsY, groupsZ} from construction.
    std::array<uint32_t, 3> defaultGroups() const noexcept
    {
        return {m_groupsX, m_groupsY, m_groupsZ};
    }
    const std::vector<CompiledShader::Binding> &bindingInfo() const noexcept
    {
        return m_bindingInfo;
    }
    bool bindingsComplete() const noexcept
    {
        return !m_descriptorSets.empty() &&
               std::all_of(m_bindingSet.begin(), m_bindingSet.end(), [](bool set) { return set; });
    }

    private:
    void init(GpuContext &ctx, const std::vector<uint32_t> &spirv,
              const std::vector<GpuBinding> &bindings, uint32_t pushConstantSize,
              uint32_t maxDescriptorSets, const std::string &entryPoint, bool reflectBindings,
              const std::vector<vk::DescriptorType> &reflectedTypes = {},
              const std::vector<CompiledShader::Binding> &bindingInfo = {},
              bool updateInitialBindings = true);

    vk::Device m_device;
    vk::UniqueDescriptorSetLayout m_descriptorSetLayout;
    vk::UniqueDescriptorPool m_descriptorPool;
    std::vector<vk::UniqueDescriptorSet> m_descriptorSets;
    vk::UniquePipelineLayout m_layout;
    vk::UniqueShaderModule m_shaderModule;
    vk::UniquePipeline m_pipeline;
    std::vector<vk::DescriptorType> m_bindingTypes;
    std::vector<CompiledShader::Binding> m_bindingInfo;
    std::vector<GpuBinding> m_bindingValues;
    std::vector<bool> m_bindingSet;
    uint32_t m_groupsX = 1, m_groupsY = 1, m_groupsZ = 1;
    uint32_t m_pushConstantSize = 0;
    GpuContext *m_ctx = nullptr;
};

} // namespace pix
