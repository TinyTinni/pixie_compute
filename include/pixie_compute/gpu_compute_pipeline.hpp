#pragma once

#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>
#include <vulkan/vulkan.hpp>

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
    /// Legacy aliases retained for source migration; use groupsX/groupsY/groupsZ.
    uint32_t workgroupX = 1;
    uint32_t workgroupY = 1;
    uint32_t workgroupZ = 1;
    uint32_t maxDescriptorSets = 16;
    std::string entryPoint = "main";
};

using PipelineDesc = GpuComputePipelineDesc;

/// Vulkan compute pipeline with automatic descriptor set management.
class GpuComputePipeline
{
    friend class GpuCommandBuffer;

    public:
    /// Create a pipeline from a GpuComputePipelineDesc config.
    GpuComputePipeline(GpuContext &ctx, const GpuComputePipelineDesc &desc);

    [[deprecated("use GpuComputePipelineDesc")]]
    GpuComputePipeline(GpuContext &ctx, const std::vector<uint32_t> &spirv,
                       const std::vector<vk::DescriptorBufferInfo> &bindings,
                       uint32_t pushConstantSize, uint32_t groupsX, uint32_t groupsY,
                       uint32_t groupsZ, uint32_t maxDescriptorSets = 16,
                       const char *entryPoint = "main");

    [[deprecated("use GpuComputePipelineDesc")]]
    GpuComputePipeline(GpuContext &ctx, const std::vector<uint32_t> &spirv,
                       const std::vector<vk::DescriptorBufferInfo> &bufferBindings,
                       const std::vector<vk::DescriptorImageInfo> &imageBindings,
                       const std::vector<vk::DescriptorType> &bindingTypes,
                       uint32_t pushConstantSize, uint32_t groupsX, uint32_t groupsY,
                       uint32_t groupsZ, uint32_t maxDescriptorSets = 16,
                       const char *entryPoint = "main");

#ifdef PIXIE_COMPUTE_HAS_SLANG
    [[deprecated("use GpuComputePipelineDesc")]]
    GpuComputePipeline(GpuContext &ctx, const std::string &slangSource,
                       const std::vector<vk::DescriptorBufferInfo> &bindings,
                       uint32_t pushConstantSize, uint32_t groupsX, uint32_t groupsY,
                       uint32_t groupsZ, uint32_t maxDescriptorSets = 16,
                       const char *entryPoint = "main");
#endif

    ~GpuComputePipeline();

    GpuComputePipeline(const GpuComputePipeline &) = delete;
    GpuComputePipeline &operator=(const GpuComputePipeline &) = delete;
    GpuComputePipeline(GpuComputePipeline &&) noexcept;
    GpuComputePipeline &operator=(GpuComputePipeline &&) noexcept;

    /// Update descriptor bindings for set 0.
    void updateBindings(const std::vector<GpuBinding> &bindings);
    /// Update descriptor bindings for a specific set.
    void updateBindings(uint32_t setIndex, const std::vector<GpuBinding> &bindings);
    [[deprecated("use flat GpuBinding list")]]
    void updateBindings(const std::vector<vk::DescriptorBufferInfo> &bufferBindings,
                        const std::vector<vk::DescriptorImageInfo> &imageBindings,
                        uint32_t setIndex = 0);
    /// Allocate and bind a new descriptor set, returning its index.
    uint32_t addDescriptorSet(const std::vector<GpuBinding> &bindings);
    /// Pre-allocate descriptor sets without binding data.
    void allocateDescriptorSets(uint32_t count);
    /// Clear all descriptor sets. Invalidates any previously returned indices from
    /// addDescriptorSet.
    void clearDescriptorSets();

    [[deprecated("use GpuCommandBuffer::bind")]]
    void bind(vk::CommandBuffer cmd) const;
    [[deprecated("use GpuCommandBuffer::bind")]]
    void bind(vk::CommandBuffer cmd, uint32_t setIndex) const;
    [[deprecated("use GpuCommandBuffer::pushConstants")]]
    void pushConstants(vk::CommandBuffer cmd, const void *data, size_t size) const;
    [[deprecated("use GpuCommandBuffer::dispatch")]]
    void dispatch(vk::CommandBuffer cmd, uint32_t groupsX, uint32_t groupsY,
                  uint32_t groupsZ) const;
    [[deprecated("use GpuCommandBuffer::dispatch")]]
    void dispatch(vk::CommandBuffer cmd) const;

    /// Assign a debug name to the pipeline (VK_EXT_debug_utils). No-op when the
    /// extension is unavailable.
    void setDebugName(const std::string &name);

    vk::Pipeline handle() const noexcept { return *m_pipeline; }
    vk::PipelineLayout layout() const noexcept { return *m_layout; }
    uint32_t descriptorSetCount() const noexcept
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

    private:
    void init(GpuContext &ctx, const std::vector<uint32_t> &spirv,
              const std::vector<GpuBinding> &bindings, uint32_t pushConstantSize,
              uint32_t maxDescriptorSets, const std::string &entryPoint, bool reflectBindings,
              const std::vector<vk::DescriptorType> &reflectedTypes = {});

    vk::Device m_device;
    vk::UniqueDescriptorSetLayout m_descriptorSetLayout;
    vk::UniqueDescriptorPool m_descriptorPool;
    std::vector<vk::UniqueDescriptorSet> m_descriptorSets;
    vk::UniquePipelineLayout m_layout;
    vk::UniqueShaderModule m_shaderModule;
    vk::UniquePipeline m_pipeline;
    std::vector<vk::DescriptorType> m_bindingTypes;
    uint32_t m_groupsX = 1, m_groupsY = 1, m_groupsZ = 1;
    uint32_t m_pushConstantSize = 0;
    GpuContext *m_ctx = nullptr;
};

} // namespace pix
