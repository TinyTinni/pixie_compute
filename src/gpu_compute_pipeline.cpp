#include "pixie_compute/gpu_compute_pipeline.hpp"

#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"

#include "error_utils.hpp"

#ifdef PIXIE_COMPUTE_HAS_SLANG
#include "pixie_compute/shader_compiler.hpp"
#endif

#include <array>
#include <unordered_map>

namespace pix
{
namespace
{

vk::UniqueShaderModule createShaderModule(vk::Device device, const std::vector<uint32_t> &spirv)
{
    vk::ShaderModuleCreateInfo info({}, spirv);
    return vkChecked([&] { return device.createShaderModuleUnique(info); },
                     "vkCreateShaderModule");
}

bool isImageType(vk::DescriptorType type)
{
    return type == vk::DescriptorType::eStorageImage ||
           type == vk::DescriptorType::eCombinedImageSampler ||
           type == vk::DescriptorType::eSampledImage;
}

bool isBufferType(vk::DescriptorType type)
{
    return type == vk::DescriptorType::eStorageBuffer ||
           type == vk::DescriptorType::eUniformBuffer ||
           type == vk::DescriptorType::eUniformTexelBuffer ||
           type == vk::DescriptorType::eStorageTexelBuffer;
}

void validateBindingType(vk::DescriptorType type)
{
    if (!isImageType(type) && !isBufferType(type))
        throw GpuError("unsupported descriptor type: " + std::to_string(static_cast<int>(type)));
}

} // namespace

GpuComputePipeline::GpuComputePipeline(GpuContext &ctx, const GpuComputePipelineDesc &desc)
    : m_device(ctx.device()), m_groupsX(desc.groupsX), m_groupsY(desc.groupsY),
      m_groupsZ(desc.groupsZ), m_ctx(&ctx)
{
    if (desc.groupsX == 1 && desc.workgroupX != 1)
        m_groupsX = desc.workgroupX;
    if (desc.groupsY == 1 && desc.workgroupY != 1)
        m_groupsY = desc.workgroupY;
    if (desc.groupsZ == 1 && desc.workgroupZ != 1)
        m_groupsZ = desc.workgroupZ;
    // Determine shader source
    std::vector<uint32_t> spirv;
    std::vector<vk::DescriptorType> reflectedTypes;
    uint32_t pushConstantSize = desc.pushConstantSize;
    bool reflectBindings = false;

    if (!desc.slangSource.empty())
    {
#ifdef PIXIE_COMPUTE_HAS_SLANG
        auto compiled = compileSlangToSpirV(desc.slangSource, desc.entryPoint);
        spirv = std::move(compiled.spirv);
        reflectedTypes = compiled.bindingTypes;
        if (pushConstantSize == 0)
            pushConstantSize = compiled.pushConstantSize;
        else if (pushConstantSize != compiled.pushConstantSize)
            throw GpuError("push constant size mismatch: shader block is " +
                           std::to_string(compiled.pushConstantSize) + " bytes, caller specified " +
                           std::to_string(pushConstantSize));
        reflectBindings = true;
#else
        throw GpuError("Slang shader compilation not available (built without PIXIE_COMPUTE_HAS_SLANG)");
#endif
    }
    else if (!desc.spirv.empty())
    {
        spirv = desc.spirv;
    }
    else
    {
        throw GpuError("GpuComputePipelineDesc must specify either spirv or slangSource");
    }

    init(ctx, spirv, desc.bindings, pushConstantSize, desc.maxDescriptorSets, desc.entryPoint,
         reflectBindings, reflectedTypes);
}

GpuComputePipeline::GpuComputePipeline(
    GpuContext &ctx, const std::vector<uint32_t> &spirv,
    const std::vector<vk::DescriptorBufferInfo> &bindings, uint32_t pushConstantSize,
    uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ, uint32_t maxDescriptorSets,
    const char *entryPoint)
    : GpuComputePipeline(ctx, [&]
                         {
                             GpuComputePipelineDesc desc;
                             desc.spirv = spirv;
                             desc.pushConstantSize = pushConstantSize;
                             desc.groupsX = groupsX;
                             desc.groupsY = groupsY;
                             desc.groupsZ = groupsZ;
                             desc.maxDescriptorSets = maxDescriptorSets;
                             desc.entryPoint = entryPoint;
                             for (const auto &binding : bindings)
                                 desc.bindings.emplace_back(binding);
                             return desc;
                         }())
{
}

GpuComputePipeline::GpuComputePipeline(
    GpuContext &ctx, const std::vector<uint32_t> &spirv,
    const std::vector<vk::DescriptorBufferInfo> &bufferBindings,
    const std::vector<vk::DescriptorImageInfo> &imageBindings,
    const std::vector<vk::DescriptorType> &bindingTypes, uint32_t pushConstantSize,
    uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ, uint32_t maxDescriptorSets,
    const char *entryPoint)
    : GpuComputePipeline(ctx, [&]
                         {
                             GpuComputePipelineDesc desc;
                             desc.spirv = spirv;
                             desc.pushConstantSize = pushConstantSize;
                             desc.groupsX = groupsX;
                             desc.groupsY = groupsY;
                             desc.groupsZ = groupsZ;
                             desc.maxDescriptorSets = maxDescriptorSets;
                             desc.entryPoint = entryPoint;
                             size_t bufferIndex = 0;
                             size_t imageIndex = 0;
                             for (auto type : bindingTypes)
                             {
                                 if (isImageType(type))
                                 {
                                     if (imageIndex >= imageBindings.size())
                                         throw GpuError("not enough image bindings");
                                     desc.bindings.emplace_back(imageBindings[imageIndex++], type);
                                 }
                                 else
                                 {
                                     if (bufferIndex >= bufferBindings.size())
                                         throw GpuError("not enough buffer bindings");
                                     desc.bindings.emplace_back(bufferBindings[bufferIndex++], type);
                                 }
                             }
                             if (bufferIndex != bufferBindings.size() ||
                                 imageIndex != imageBindings.size())
                                 throw GpuError("descriptor binding count mismatch");
                             return desc;
                         }())
{
}

#ifdef PIXIE_COMPUTE_HAS_SLANG
GpuComputePipeline::GpuComputePipeline(
    GpuContext &ctx, const std::string &slangSource,
    const std::vector<vk::DescriptorBufferInfo> &bindings, uint32_t pushConstantSize,
    uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ, uint32_t maxDescriptorSets,
    const char *entryPoint)
    : GpuComputePipeline(ctx, [&]
                         {
                             GpuComputePipelineDesc desc;
                             desc.slangSource = slangSource;
                             desc.pushConstantSize = pushConstantSize;
                             desc.groupsX = groupsX;
                             desc.groupsY = groupsY;
                             desc.groupsZ = groupsZ;
                             desc.maxDescriptorSets = maxDescriptorSets;
                             desc.entryPoint = entryPoint;
                             for (const auto &binding : bindings)
                                 desc.bindings.emplace_back(binding);
                             return desc;
                         }())
{
}
#endif

void GpuComputePipeline::init(GpuContext &ctx, const std::vector<uint32_t> &spirv,
                              const std::vector<GpuBinding> &bindings, uint32_t pushConstantSize,
                              uint32_t maxDescriptorSets, const std::string &entryPoint,
                              bool reflectBindings,
                              const std::vector<vk::DescriptorType> &reflectedTypes)
{
    auto &device = m_device;
    m_pushConstantSize = pushConstantSize;
    m_shaderModule = createShaderModule(device, spirv);

    uint32_t n = static_cast<uint32_t>(bindings.size());

    // Build binding types vector (validate or use reflection)
    const bool haveReflectedTypes = reflectBindings && !reflectedTypes.empty();
    if (haveReflectedTypes && bindings.size() != reflectedTypes.size())
        throw GpuError("binding count mismatch: shader has " +
                       std::to_string(reflectedTypes.size()) + " bindings, got " +
                       std::to_string(bindings.size()));
    m_bindingTypes.resize(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        const auto &binding = bindings[i];
        bool isBuffer = std::holds_alternative<vk::DescriptorBufferInfo>(binding.resource);
        bool isImage = std::holds_alternative<vk::DescriptorImageInfo>(binding.resource);

        if (haveReflectedTypes)
        {
#ifdef PIXIE_COMPUTE_HAS_SLANG
            vk::DescriptorType reflectedType = reflectedTypes[i];

            // Validate variant kind matches reflection
            if (isBuffer && isImageType(reflectedType))
                throw GpuError("binding " + std::to_string(i) +
                               ": buffer provided but shader expects image");
            if (isImage && isBufferType(reflectedType))
                throw GpuError("binding " + std::to_string(i) +
                               ": image provided but shader expects buffer");

            m_bindingTypes[i] = reflectedType;
#else
            throw GpuError("reflection not available");
#endif
        }
        else
        {
            // SPIR-V path: use provided type, validate
            validateBindingType(binding.type);
            if (isBuffer && !isBufferType(binding.type))
                throw GpuError("binding " + std::to_string(i) +
                               ": buffer provided but type is image type");
            if (isImage && !isImageType(binding.type))
                throw GpuError("binding " + std::to_string(i) +
                               ": image provided but type is buffer type");
            m_bindingTypes[i] = binding.type;
        }
    }

    // Create descriptor set layout
    std::vector<vk::DescriptorSetLayoutBinding> dslBindings;
    std::vector<vk::DescriptorBindingFlags> flags;
    dslBindings.reserve(n);
    flags.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        dslBindings.push_back({i, m_bindingTypes[i], 1, vk::ShaderStageFlagBits::eCompute});
        flags.push_back(vk::DescriptorBindingFlagBits::eUpdateAfterBind);
    }
    vk::DescriptorSetLayoutBindingFlagsCreateInfo flagInfo(flags);
    vk::DescriptorSetLayoutCreateInfo dslInfo(
        vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, dslBindings);
    dslInfo.setPNext(&flagInfo);
    m_descriptorSetLayout = vkChecked([&] { return device.createDescriptorSetLayoutUnique(dslInfo); },
                                      "vkCreateDescriptorSetLayout");

    // Create descriptor pool
    std::unordered_map<vk::DescriptorType, uint32_t> typeCounts;
    for (auto t : m_bindingTypes)
        typeCounts[t]++;

    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.reserve(typeCounts.size());
    for (auto &[type, count] : typeCounts)
        poolSizes.push_back({type, maxDescriptorSets * count});

    vk::DescriptorPoolCreateInfo poolInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
            vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        maxDescriptorSets, static_cast<uint32_t>(poolSizes.size()), poolSizes.data());
    m_descriptorPool = vkChecked([&] { return device.createDescriptorPoolUnique(poolInfo); },
                                 "vkCreateDescriptorPool");

    // Allocate and update set 0
    allocateDescriptorSets(1);
    updateBindings(bindings);

    // Create pipeline layout
    vk::PushConstantRange pcRange(vk::ShaderStageFlagBits::eCompute, 0, pushConstantSize);
    vk::PipelineLayoutCreateInfo plInfo({}, 1, &*m_descriptorSetLayout,
                                        pushConstantSize > 0 ? 1u : 0u, &pcRange);
    m_layout = vkChecked([&] { return device.createPipelineLayoutUnique(plInfo); },
                         "vkCreatePipelineLayout");

    // Create pipeline
    vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, *m_shaderModule,
                                            entryPoint.c_str());
    vk::ComputePipelineCreateInfo cpInfo({}, stage, *m_layout);
    auto result = device.createComputePipelineUnique(ctx.pipelineCache(), cpInfo);
    if (result.result != vk::Result::eSuccess)
        throw GpuError("compute pipeline creation failed (VkResult " +
                       std::to_string(static_cast<int>(result.result)) + ")");
    m_pipeline = std::move(result.value);
}

GpuComputePipeline::~GpuComputePipeline() = default;

GpuComputePipeline::GpuComputePipeline(GpuComputePipeline &&other) noexcept
    : m_device(other.m_device), m_descriptorSetLayout(std::move(other.m_descriptorSetLayout)),
      m_descriptorPool(std::move(other.m_descriptorPool)),
      m_descriptorSets(std::move(other.m_descriptorSets)), m_layout(std::move(other.m_layout)),
      m_shaderModule(std::move(other.m_shaderModule)), m_pipeline(std::move(other.m_pipeline)),
      m_bindingTypes(std::move(other.m_bindingTypes)), m_groupsX(other.m_groupsX),
      m_groupsY(other.m_groupsY), m_groupsZ(other.m_groupsZ),
      m_pushConstantSize(other.m_pushConstantSize), m_ctx(other.m_ctx)
{
    other.m_ctx = nullptr;
}

GpuComputePipeline &GpuComputePipeline::operator=(GpuComputePipeline &&other) noexcept
{
    if (this != &other)
    {
        m_device = other.m_device;
        m_descriptorSetLayout = std::move(other.m_descriptorSetLayout);
        m_descriptorPool = std::move(other.m_descriptorPool);
        m_descriptorSets = std::move(other.m_descriptorSets);
        m_layout = std::move(other.m_layout);
        m_shaderModule = std::move(other.m_shaderModule);
        m_pipeline = std::move(other.m_pipeline);
        m_bindingTypes = std::move(other.m_bindingTypes);
        m_groupsX = other.m_groupsX;
        m_groupsY = other.m_groupsY;
        m_groupsZ = other.m_groupsZ;
        m_pushConstantSize = other.m_pushConstantSize;
        m_ctx = other.m_ctx;
        other.m_ctx = nullptr;
    }
    return *this;
}

void GpuComputePipeline::setDebugName(const std::string &name)
{
    if (!m_ctx)
        return;
    m_ctx->setDebugName(*m_pipeline, name);
}

void GpuComputePipeline::updateBindings(const std::vector<GpuBinding> &bindings)
{
    updateBindings(0, bindings);
}

void GpuComputePipeline::updateBindings(
    const std::vector<vk::DescriptorBufferInfo> &bufferBindings,
    const std::vector<vk::DescriptorImageInfo> &imageBindings, uint32_t setIndex)
{
    std::vector<GpuBinding> flat;
    size_t bufferIndex = 0;
    size_t imageIndex = 0;
    flat.reserve(m_bindingTypes.size());
    for (auto type : m_bindingTypes)
    {
        if (isImageType(type))
        {
            if (imageIndex >= imageBindings.size())
                throw GpuError("not enough image bindings");
            flat.emplace_back(imageBindings[imageIndex++], type);
        }
        else
        {
            if (bufferIndex >= bufferBindings.size())
                throw GpuError("not enough buffer bindings");
            flat.emplace_back(bufferBindings[bufferIndex++], type);
        }
    }
    if (bufferIndex != bufferBindings.size() || imageIndex != imageBindings.size())
        throw GpuError("descriptor binding count mismatch");
    updateBindings(setIndex, flat);
}

void GpuComputePipeline::bind(vk::CommandBuffer cmd) const
{
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *m_pipeline);
    if (!m_descriptorSets.empty())
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_layout, 0,
                               *m_descriptorSets[0], {});
}

void GpuComputePipeline::bind(vk::CommandBuffer cmd, uint32_t setIndex) const
{
    if (setIndex >= m_descriptorSets.size())
        throw GpuError("bind: setIndex exceeds descriptor set count");
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *m_pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_layout, 0,
                           *m_descriptorSets[setIndex], {});
}

void GpuComputePipeline::pushConstants(vk::CommandBuffer cmd, const void *data, size_t size) const
{
    if (size > m_pushConstantSize)
        throw GpuError("push constant size exceeds pipeline layout range");
    cmd.pushConstants(*m_layout, vk::ShaderStageFlagBits::eCompute, 0,
                      static_cast<uint32_t>(size), data);
}

void GpuComputePipeline::dispatch(vk::CommandBuffer cmd, uint32_t groupsX, uint32_t groupsY,
                                  uint32_t groupsZ) const
{
    cmd.dispatch(groupsX, groupsY, groupsZ);
}

void GpuComputePipeline::dispatch(vk::CommandBuffer cmd) const
{
    cmd.dispatch(m_groupsX, m_groupsY, m_groupsZ);
}

void GpuComputePipeline::updateBindings(uint32_t setIndex, const std::vector<GpuBinding> &bindings)
{
    if (bindings.size() != m_bindingTypes.size())
        throw GpuError("updateBindings: expected " + std::to_string(m_bindingTypes.size()) +
                       " bindings, got " + std::to_string(bindings.size()));
    if (setIndex >= static_cast<uint32_t>(m_descriptorSets.size()))
        allocateDescriptorSets(setIndex + 1 - static_cast<uint32_t>(m_descriptorSets.size()));

    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(bindings.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(bindings.size()); ++i)
    {
        const auto &binding = bindings[i];
        if (isImageType(m_bindingTypes[i]))
        {
            if (!std::holds_alternative<vk::DescriptorImageInfo>(binding.resource))
                throw GpuError("binding " + std::to_string(i) + ": expected image, got buffer");
            const auto &imgInfo = std::get<vk::DescriptorImageInfo>(binding.resource);
            writes.push_back({*m_descriptorSets[setIndex], i, 0, 1, m_bindingTypes[i], &imgInfo});
        }
        else
        {
            if (!std::holds_alternative<vk::DescriptorBufferInfo>(binding.resource))
                throw GpuError("binding " + std::to_string(i) + ": expected buffer, got image");
            const auto &bufInfo = std::get<vk::DescriptorBufferInfo>(binding.resource);
            writes.push_back({*m_descriptorSets[setIndex], i, 0, 1, m_bindingTypes[i], nullptr,
                              &bufInfo});
        }
    }
    m_device.updateDescriptorSets(writes, {});
}

uint32_t GpuComputePipeline::addDescriptorSet(const std::vector<GpuBinding> &bindings)
{
    uint32_t idx = static_cast<uint32_t>(m_descriptorSets.size());
    allocateDescriptorSets(1);
    updateBindings(idx, bindings);
    return idx;
}

void GpuComputePipeline::allocateDescriptorSets(uint32_t count)
{
    if (count == 0)
        return;
    std::vector<vk::DescriptorSetLayout> layouts(count, *m_descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo(*m_descriptorPool,
                                            static_cast<uint32_t>(layouts.size()), layouts.data());
    auto sets = vkChecked([&] { return m_device.allocateDescriptorSetsUnique(allocInfo); },
                          "vkAllocateDescriptorSets");
    for (auto &s : sets)
        m_descriptorSets.push_back(std::move(s));
}

void GpuComputePipeline::clearDescriptorSets()
{
    m_descriptorSets.clear();
}

} // namespace pix
