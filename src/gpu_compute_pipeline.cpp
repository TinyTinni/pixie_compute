#include "pixie_compute/gpu_compute_pipeline.hpp"

#include "error_utils.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"

#ifdef PIXIE_COMPUTE_HAS_SLANG
#include "pixie_compute/shader_compiler.hpp"
#endif

#include <algorithm>
#include <array>
#include <unordered_map>

namespace pix
{
namespace
{

vk::UniqueShaderModule createShaderModule(vk::Device device, const std::vector<uint32_t> &spirv)
{
    vk::ShaderModuleCreateInfo info({}, spirv);
    return vkChecked([&] { return device.createShaderModuleUnique(info); }, "vkCreateShaderModule");
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
    // Determine shader source
    std::vector<uint32_t> spirv;
    std::vector<vk::DescriptorType> reflectedTypes;
    std::vector<CompiledShader::Binding> reflectedBindings;
    std::vector<GpuBinding> bindings = desc.bindings;
    uint32_t pushConstantSize = desc.pushConstantSize;
    bool reflectBindings = false;

    if (!desc.slangSource.empty())
    {
#ifdef PIXIE_COMPUTE_HAS_SLANG
        auto compiled = compileSlangToSpirV(desc.slangSource, desc.entryPoint);
        spirv = std::move(compiled.spirv);
        reflectedTypes = compiled.bindingTypes;
        reflectedBindings = std::move(compiled.bindings);
        if (pushConstantSize == 0)
            pushConstantSize = compiled.pushConstantSize;
        else if (pushConstantSize != compiled.pushConstantSize)
            throw GpuError("push constant size mismatch: shader block is " +
                           std::to_string(compiled.pushConstantSize) + " bytes, caller specified " +
                           std::to_string(pushConstantSize));
        reflectBindings = true;
        if (bindings.empty() && !reflectedTypes.empty())
        {
            bindings.reserve(reflectedTypes.size());
            for (auto type : reflectedTypes)
            {
                if (isImageType(type))
                    bindings.emplace_back(vk::DescriptorImageInfo{}, type);
                else
                    bindings.emplace_back(vk::DescriptorBufferInfo{}, type);
            }
        }
#else
        throw GpuError(
            "Slang shader compilation not available (built without PIXIE_COMPUTE_HAS_SLANG)");
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

    init(ctx, spirv, bindings, pushConstantSize, desc.maxDescriptorSets, desc.entryPoint,
         reflectBindings, reflectedTypes, reflectedBindings, !desc.bindings.empty());
}

void GpuComputePipeline::init(GpuContext &ctx, const std::vector<uint32_t> &spirv,
                              const std::vector<GpuBinding> &bindings, uint32_t pushConstantSize,
                              uint32_t maxDescriptorSets, const std::string &entryPoint,
                              bool reflectBindings,
                              const std::vector<vk::DescriptorType> &reflectedTypes,
                              const std::vector<CompiledShader::Binding> &bindingInfo,
                              bool updateInitialBindings)
{
    auto &device = m_device;
    m_pushConstantSize = pushConstantSize;
    m_bindingInfo = bindingInfo;
    m_bindingValues = bindings;
    m_bindingSet.assign(bindings.size(), updateInitialBindings);
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
    m_descriptorSetLayout =
        vkChecked([&] { return device.createDescriptorSetLayoutUnique(dslInfo); },
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
    allocateBindingSets(1);
    if (updateInitialBindings)
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
      m_bindingTypes(std::move(other.m_bindingTypes)),
      m_bindingInfo(std::move(other.m_bindingInfo)),
      m_bindingValues(std::move(other.m_bindingValues)),
      m_bindingSet(std::move(other.m_bindingSet)), m_groupsX(other.m_groupsX),
      m_groupsY(other.m_groupsY), m_groupsZ(other.m_groupsZ),
      m_pushConstantSize(other.m_pushConstantSize), m_ctx(other.m_ctx)
{
    other.m_ctx = nullptr;
}

GpuComputePipeline &GpuComputePipeline::operator=(GpuComputePipeline &&other) noexcept
{
    if (this != &other)
    {
        // Swap states: the moved-from object now owns whatever this pipeline held
        // and its destructor releases it.
        using std::swap;
        swap(m_device, other.m_device);
        swap(m_descriptorSetLayout, other.m_descriptorSetLayout);
        swap(m_descriptorPool, other.m_descriptorPool);
        swap(m_descriptorSets, other.m_descriptorSets);
        swap(m_layout, other.m_layout);
        swap(m_shaderModule, other.m_shaderModule);
        swap(m_pipeline, other.m_pipeline);
        swap(m_bindingTypes, other.m_bindingTypes);
        swap(m_bindingInfo, other.m_bindingInfo);
        swap(m_bindingValues, other.m_bindingValues);
        swap(m_bindingSet, other.m_bindingSet);
        swap(m_groupsX, other.m_groupsX);
        swap(m_groupsY, other.m_groupsY);
        swap(m_groupsZ, other.m_groupsZ);
        swap(m_pushConstantSize, other.m_pushConstantSize);
        swap(m_ctx, other.m_ctx);
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
    m_bindingValues = bindings;
    m_bindingSet.assign(bindings.size(), true);
    updateBindingSet(0, bindings);
}

void GpuComputePipeline::updateBinding(std::string_view name, const GpuBinding &binding)
{
    auto it = std::find_if(m_bindingInfo.begin(), m_bindingInfo.end(),
                           [&](const auto &info) { return info.name == name && info.set == 0; });
    if (it == m_bindingInfo.end())
        throw GpuError("shader binding not found: " + std::string(name));
    if (m_descriptorSets.empty())
        throw GpuError("no descriptor sets available to update: " + std::string(name));
    const uint32_t index = it->binding;
    if (index >= m_bindingValues.size())
        throw GpuError("shader binding index out of range: " + std::string(name));
    if (binding.type != m_bindingTypes[index])
        throw GpuError("descriptor type mismatch for shader binding: " + std::string(name));
    if (isImageType(m_bindingTypes[index]) !=
        std::holds_alternative<vk::DescriptorImageInfo>(binding.resource))
        throw GpuError("descriptor resource kind mismatch for shader binding: " +
                       std::string(name));
    m_bindingValues[index] = binding;
    m_bindingSet[index] = true;
    if (isImageType(m_bindingTypes[index]))
    {
        const auto &info = std::get<vk::DescriptorImageInfo>(binding.resource);
        m_device.updateDescriptorSets(
            {{*m_descriptorSets[0], index, 0, 1, m_bindingTypes[index], &info}}, {});
    }
    else
    {
        const auto &info = std::get<vk::DescriptorBufferInfo>(binding.resource);
        m_device.updateDescriptorSets(
            {{*m_descriptorSets[0], index, 0, 1, m_bindingTypes[index], nullptr, &info}}, {});
    }
}

void GpuComputePipeline::updateBindingSet(uint32_t bindingSetIndex,
                                          const std::vector<GpuBinding> &bindings)
{
    if (bindings.size() != m_bindingTypes.size())
        throw GpuError("updateBindingSet: expected " + std::to_string(m_bindingTypes.size()) +
                       " bindings, got " + std::to_string(bindings.size()));
    if (bindingSetIndex >= static_cast<uint32_t>(m_descriptorSets.size()))
        allocateBindingSets(bindingSetIndex + 1 - static_cast<uint32_t>(m_descriptorSets.size()));

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
            writes.push_back(
                {*m_descriptorSets[bindingSetIndex], i, 0, 1, m_bindingTypes[i], &imgInfo});
        }
        else
        {
            if (!std::holds_alternative<vk::DescriptorBufferInfo>(binding.resource))
                throw GpuError("binding " + std::to_string(i) + ": expected buffer, got image");
            const auto &bufInfo = std::get<vk::DescriptorBufferInfo>(binding.resource);
            writes.push_back({*m_descriptorSets[bindingSetIndex], i, 0, 1, m_bindingTypes[i],
                              nullptr, &bufInfo});
        }
    }
    m_device.updateDescriptorSets(writes, {});
}

uint32_t GpuComputePipeline::addBindingSet(const std::vector<GpuBinding> &bindings)
{
    uint32_t idx = static_cast<uint32_t>(m_descriptorSets.size());
    allocateBindingSets(1);
    updateBindingSet(idx, bindings);
    return idx;
}

void GpuComputePipeline::allocateBindingSets(uint32_t count)
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

void GpuComputePipeline::clearBindingSets()
{
    m_descriptorSets.clear();
}

} // namespace pix
