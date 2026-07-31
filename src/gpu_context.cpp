#include "pixie_compute/gpu_context.hpp"

#include "error_utils.hpp"
#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_timeline_semaphore.hpp"
#include "pixie_compute/utility.hpp"

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG_VVL "pixie_vvl"
#endif

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace pix
{
namespace
{

static std::string formatBytes(uint64_t bytes)
{
    static const char *const units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    unsigned int unit = 0;
    while (value >= 1024.0 && unit < 3)
    {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(unit == 0 ? 0 : 1) << value << " " << units[unit];
    return os.str();
}

static void breakOnValidationError()
{
#if defined(_WIN32)
    __debugbreak();
#else
    std::raise(SIGTRAP);
#endif
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT /*type*/,
    const vk::DebugUtilsMessengerCallbackDataEXT *data, void *userData)
{
    auto *desc = static_cast<const GpuContextDesc *>(userData);
    if (desc && desc->breakOnError && severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
    {
        breakOnValidationError();
    }
    const bool isShaderPrintf = data && data->pMessageIdName &&
                                std::strstr(data->pMessageIdName, "DEBUG-PRINTF") != nullptr;
    if (isShaderPrintf && desc && desc->shaderPrintfSink && data)
    {
        desc->shaderPrintfSink(data->pMessage ? data->pMessage : "");
        return VK_FALSE;
    }
    if (desc && desc->logSink && data)
    {
        desc->logSink(severity, data->pMessage ? data->pMessage : "");
        return VK_FALSE;
    }
    if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
    {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG_VVL, "[VVL] %s", data ? data->pMessage : "");
#else
        std::cerr << "[VVL] " << (data ? data->pMessage : "") << std::endl;
#endif
    }
    return VK_FALSE;
}

bool hasInstanceLayer(const char *name)
{
    auto props = vk::enumerateInstanceLayerProperties();
    return std::any_of(props.begin(), props.end(), [&](const vk::LayerProperties &p)
                       { return std::strcmp(p.layerName, name) == 0; });
}

bool hasInstanceExtension(const char *name)
{
    auto props = vk::enumerateInstanceExtensionProperties();
    return std::any_of(props.begin(), props.end(), [&](const vk::ExtensionProperties &p)
                       { return std::strcmp(p.extensionName, name) == 0; });
}

struct InstanceResult
{
    vk::UniqueInstance instance;
    bool validationEnabled = false;
    bool debugUtilsEnabled = false;
    bool shaderPrintfValidationEnabled = false;
    std::vector<std::string> instanceExtensions;
};

InstanceResult createInstance(const std::vector<std::string> &extraExtensions,
                              bool enableValidation, bool enableShaderPrintf, uint32_t apiVersion)
{
    if (volkInitialize() != VK_SUCCESS)
        throw GpuError("volkInitialize failed - Vulkan not available");

    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    vk::ApplicationInfo appInfo("pixie_compute", 0, "pixie_compute", 0, apiVersion);

    std::vector<const char *> layers;
    std::vector<const char *> extensions;
    std::vector<vk::ValidationFeatureEnableEXT> validationFeatures;
    bool shaderPrintfValidationEnabled = false;
    bool validationEnabled = false;

    if (enableValidation)
    {
        if (hasInstanceLayer("VK_LAYER_KHRONOS_validation") &&
            hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
        {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            validationEnabled = true;
            if (enableShaderPrintf &&
                hasInstanceExtension(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME))
            {
                extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
                validationFeatures.push_back(vk::ValidationFeatureEnableEXT::eDebugPrintf);
                shaderPrintfValidationEnabled = true;
            }
        }
        else
        {
            std::cerr << "[VVL] validation requested but layer/extension unavailable; skipping"
                      << std::endl;
        }
    }
    // Enable VK_EXT_debug_utils whenever it is available, not only with validation, so
    // object names and command labels work in debuggers and profilers without the
    // validation layer.
    if (hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
    {
        if (std::find(extensions.begin(), extensions.end(), VK_EXT_DEBUG_UTILS_EXTENSION_NAME) ==
            extensions.end())
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }
    for (auto &ext : extraExtensions)
        extensions.push_back(ext.c_str());

    vk::InstanceCreateInfo info({}, &appInfo, static_cast<uint32_t>(layers.size()), layers.data(),
                                static_cast<uint32_t>(extensions.size()), extensions.data());
    vk::ValidationFeaturesEXT validationInfo;
    if (!validationFeatures.empty())
    {
        validationInfo.setEnabledValidationFeatures(validationFeatures);
        info.setPNext(&validationInfo);
    }
    InstanceResult result{
        vkChecked([&] { return vk::createInstanceUnique(info); }, "vkCreateInstance"),
        validationEnabled,
        hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME),
        shaderPrintfValidationEnabled,
        {}};
    for (auto *ext : extensions)
        result.instanceExtensions.emplace_back(ext);
    return result;
}

vk::UniqueDebugUtilsMessengerEXT
createDebugMessenger(vk::Instance instance, vk::DebugUtilsMessageSeverityFlagsEXT severity,
                     void *userData)
{
    vk::DebugUtilsMessengerCreateInfoEXT dbgInfo{};
    dbgInfo.messageSeverity = severity;
    dbgInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                          vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                          vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                          vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding;
    // The callback field type differs between vulkan.hpp generations (raw C
    // PFN_vkDebugUtilsMessengerCallbackEXT vs the vk-namespace alias). The
    // vk::-typed callback is ABI-compatible with both; cast to the field's
    // declared type. vulkan.hpp itself uses the same cast with this pragma.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
    dbgInfo.pfnUserCallback = reinterpret_cast<decltype(dbgInfo.pfnUserCallback)>(&debugCallback);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    dbgInfo.pUserData = userData;
    return instance.createDebugUtilsMessengerEXTUnique(dbgInfo);
}

bool effectiveValidation(const GpuContextDesc &desc)
{
    bool result = desc.enableValidation;
    if (const char *env = std::getenv("PIXIE_COMPUTE_VALIDATION"))
    {
        std::string value(env);
        if (value == "0" || value == "false" || value == "off")
            result = false;
        else if (value == "1" || value == "true" || value == "on")
            result = true;
    }
    return result;
}

// Link a list of feature structs into a single pNext chain, in order, by
// overwriting each struct's pNext (the second member of every Vulkan structure).
// Returns the head pointer, or nullptr when the list is empty.
void *linkFeatureChain(const std::vector<const void *> &features)
{
    for (size_t i = 0; i < features.size(); ++i)
    {
        auto *base = static_cast<VkBaseInStructure *>(const_cast<void *>(features[i]));
        base->pNext = (i + 1 < features.size())
                          ? static_cast<const VkBaseInStructure *>(features[i + 1])
                          : nullptr;
    }
    return features.empty() ? nullptr : const_cast<void *>(features.front());
}

} // namespace

GpuContext::GpuContext(const GpuContextDesc &desc) : m_desc(desc)
{
    auto result = createInstance(desc.instanceExtensions, effectiveValidation(desc),
                                 desc.enableShaderPrintf, desc.vulkanApiVersion);
    m_instance = std::move(result.instance);
    m_validation = result.validationEnabled;
    m_debugUtilsEnabled = result.debugUtilsEnabled;
    const bool shaderPrintfValidationEnabled = result.shaderPrintfValidationEnabled;
    m_instanceExtensions = std::move(result.instanceExtensions);

    volkLoadInstance(*m_instance);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_instance, vkGetInstanceProcAddr);
    if (m_validation)
    {
        auto severity = m_desc.debugSeverity;
        if (m_desc.enableShaderPrintf)
            severity |= vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo;
        m_debugMessenger = createDebugMessenger(*m_instance, severity, &m_desc);
    }

    selectPhysicalDevice();
    findQueueFamily();
    findTransferQueueFamily();
    createDevice();
    m_shaderPrintfEnabled =
        m_desc.enableShaderPrintf && shaderPrintfValidationEnabled &&
        std::any_of(m_deviceExtensions.begin(), m_deviceExtensions.end(), [](const std::string &ext)
                    { return ext == VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME; });
    createCommandPool();
    createAllocator();
    createPipelineCache();

#ifdef __ANDROID__
    auto props = m_physicalDevice.getProperties();
    __android_log_print(ANDROID_LOG_INFO, "pixie_gpu", "Vulkan device: %s (type=%d)",
                        props.deviceName.data(), static_cast<int>(props.deviceType));

    auto extensions = m_physicalDevice.enumerateDeviceExtensionProperties();
    for (auto &ext : extensions)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "pixie_gpu", "  ext: %s v%u", ext.extensionName,
                            ext.specVersion);
    }

    auto queueProps = m_physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "pixie_gpu", "  queue[%u] flags=0x%x count=%u", i,
                            static_cast<uint32_t>(queueProps[i].queueFlags),
                            queueProps[i].queueCount);
    }
#endif
}

void GpuContext::selectPhysicalDevice()
{
    auto devices = m_instance->enumeratePhysicalDevices();
    if (devices.empty())
        throw GpuError("no Vulkan physical devices found");

    if (m_desc.deviceScore)
    {
        uint32_t bestScore = 0;
        vk::PhysicalDevice bestDevice = devices[0];
        for (auto &dev : devices)
        {
            auto props = dev.getProperties();
            uint32_t score = m_desc.deviceScore(dev, props);
            if (score > bestScore)
            {
                bestScore = score;
                bestDevice = dev;
            }
        }
        m_physicalDevice = bestDevice;
    }
    else
    {
        std::sort(devices.begin(), devices.end(),
                  [](const vk::PhysicalDevice &a, const vk::PhysicalDevice &b)
                  {
                      auto pa = a.getProperties();
                      auto pb = b.getProperties();
                      if (pa.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
                          pb.deviceType != vk::PhysicalDeviceType::eDiscreteGpu)
                          return true;
                      if (pb.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
                          pa.deviceType != vk::PhysicalDeviceType::eDiscreteGpu)
                          return false;
                      return pa.limits.maxComputeWorkGroupCount[0] >
                             pb.limits.maxComputeWorkGroupCount[0];
                  });
        m_physicalDevice = devices[0];
    }
}

void GpuContext::findQueueFamily()
{
    auto queueProps = m_physicalDevice.getQueueFamilyProperties();

    auto hasFlags = [&](uint32_t i, vk::QueueFlags flags)
    { return (queueProps[i].queueFlags & flags) == flags; };

    vk::QueueFlags required = vk::QueueFlagBits::eCompute;
    if (m_desc.requireGraphics)
        required |= vk::QueueFlagBits::eGraphics;

    int bestScore = -1;
    uint32_t bestIndex = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i)
    {
        if (!hasFlags(i, required))
            continue;
        int score = 0;
        if (m_desc.surface)
        {
            VkBool32 present = VK_FALSE;
            auto func = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
                vkGetInstanceProcAddr(*m_instance, "vkGetPhysicalDeviceSurfaceSupportKHR"));
            if (func)
                func(m_physicalDevice, i, m_desc.surface, &present);
            if (present)
                score += 2;
        }
        if (hasFlags(i, vk::QueueFlagBits::eGraphics))
            score += 1;
        if (score > bestScore)
        {
            bestScore = score;
            bestIndex = i;
        }
    }
    if (bestScore < 0)
        throw GpuError("no suitable queue family found");
    m_queueFamilyIndex = bestIndex;
}

void GpuContext::findTransferQueueFamily()
{
    auto queueProps = m_physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i)
    {
        auto flags = queueProps[i].queueFlags;
        bool hasTransfer = (flags & vk::QueueFlagBits::eTransfer) != vk::QueueFlags{};
        bool hasCompute = (flags & vk::QueueFlagBits::eCompute) != vk::QueueFlags{};
        bool hasGraphics = (flags & vk::QueueFlagBits::eGraphics) != vk::QueueFlags{};

        if (hasTransfer && !hasCompute && !hasGraphics)
        {
            m_transferQueueFamilyIndex = i;
            m_hasSeparateTransferQueue = true;
            return;
        }
    }

    // No dedicated transfer queue found — fall back to compute queue
    m_transferQueueFamilyIndex = m_queueFamilyIndex;
    m_hasSeparateTransferQueue = false;
}

void GpuContext::createDevice()
{
    float queuePriority = 1.0f;

    std::vector<vk::DeviceQueueCreateInfo> queueInfos;
    queueInfos.emplace_back(vk::DeviceQueueCreateFlags{}, m_queueFamilyIndex, 1, &queuePriority);

    if (m_hasSeparateTransferQueue && m_transferQueueFamilyIndex != m_queueFamilyIndex)
    {
        queueInfos.emplace_back(vk::DeviceQueueCreateFlags{}, m_transferQueueFamilyIndex, 1,
                                &queuePriority);
    }

    auto available = m_physicalDevice.enumerateDeviceExtensionProperties();

    // Extensions the library requires internally. Descriptor indexing and timeline
    // semaphores are core since Vulkan 1.2, where they must not be requested by name.
    std::vector<const char *> deviceExtensions;
    if (m_desc.vulkanApiVersion < VK_API_VERSION_1_2)
    {
        deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }
    if (m_desc.surface)
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    // Enable VK_EXT_debug_utils on the device when the instance has it, so object
    // naming and command labels take effect even without the validation layer.
    if (m_debugUtilsEnabled)
    {
        bool devAvailable = std::any_of(
            available.begin(), available.end(), [](const vk::ExtensionProperties &p)
            { return std::strcmp(p.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0; });
        if (devAvailable)
            deviceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        else
            m_debugUtilsEnabled = false;
    }
    if (m_desc.enableShaderPrintf)
    {
        const bool availableNonSemantic =
            std::any_of(available.begin(), available.end(),
                        [](const vk::ExtensionProperties &p) {
                            return std::strcmp(p.extensionName,
                                               VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME) == 0;
                        });
        if (!availableNonSemantic)
            throw GpuError(
                "shader printf requested but VK_KHR_shader_non_semantic_info is unavailable");
        deviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }
    if (m_desc.enableDeviceFaultDiagnostics)
    {
        const bool availableDeviceFault = std::any_of(
            available.begin(), available.end(), [](const vk::ExtensionProperties &p)
            { return std::strcmp(p.extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME) == 0; });
        if (availableDeviceFault)
            deviceExtensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
    }

    std::vector<const char *> enabled;
    for (auto *ext : deviceExtensions)
    {
        auto it =
            std::find_if(available.begin(), available.end(), [ext](const vk::ExtensionProperties &p)
                         { return std::strcmp(p.extensionName, ext) == 0; });
        if (it != available.end())
        {
            enabled.push_back(ext);
        }
        else
        {
            throw GpuError(std::string("required device extension not available: ") + ext);
        }
    }

    // User-requested extensions: an onMissing callback decides whether a missing
    // extension is skipped (returns true) or aborts creation (false or no callback).
    for (const auto &ext : m_desc.deviceExtensions)
    {
        auto it = std::find_if(available.begin(), available.end(),
                               [&ext](const vk::ExtensionProperties &p)
                               { return std::strcmp(p.extensionName, ext.name) == 0; });
        if (it != available.end())
        {
            enabled.push_back(ext.name);
        }
        else if (ext.onMissing && ext.onMissing(ext.name))
        {
            m_skippedDeviceExtensions.emplace_back(ext.name);
        }
        else
        {
            throw GpuError(std::string("required device extension not available: ") +
                           (ext.name ? ext.name : "<unnamed>"));
        }
    }

    // Record which extensions were enabled (for infoDump()) and confirm the device-level
    // debug-utils state.
    m_deviceExtensions.clear();
    for (auto *ext : enabled)
        m_deviceExtensions.emplace_back(ext);
    m_debugUtilsEnabled =
        std::any_of(enabled.begin(), enabled.end(), [](const char *ext)
                    { return std::strcmp(ext, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0; });
    m_deviceFaultEnabled =
        m_desc.enableDeviceFaultDiagnostics &&
        std::any_of(enabled.begin(), enabled.end(), [](const char *ext)
                    { return std::strcmp(ext, VK_EXT_DEVICE_FAULT_EXTENSION_NAME) == 0; });

    // Query physical device feature support for the internally required features.
    vk::PhysicalDeviceFeatures2 features2;
    vk::PhysicalDeviceDescriptorIndexingFeatures supportedIndexingFeatures;
    vk::PhysicalDeviceTimelineSemaphoreFeatures supportedTimelineFeatures;
    features2.pNext = &supportedIndexingFeatures;
    supportedIndexingFeatures.pNext = &supportedTimelineFeatures;
    m_physicalDevice.getFeatures2(&features2);

    // Check required features
    if (!supportedTimelineFeatures.timelineSemaphore)
        throw GpuError("timelineSemaphore not supported");
    if (!supportedIndexingFeatures.descriptorBindingStorageBufferUpdateAfterBind)
        throw GpuError("descriptorBindingStorageBufferUpdateAfterBind not supported");
    if (!supportedIndexingFeatures.descriptorBindingStorageImageUpdateAfterBind)
        throw GpuError("descriptorBindingStorageImageUpdateAfterBind not supported");
    if (!supportedIndexingFeatures.descriptorBindingSampledImageUpdateAfterBind)
        throw GpuError("descriptorBindingSampledImageUpdateAfterBind not supported");

    // Validate user-requested core features against supported
    vk::PhysicalDeviceFeatures supportedCore = features2.features;
    const auto &requested = m_desc.coreFeatures;
    std::vector<std::string> unsupportedFeatures;
#define CHECK_FEATURE(name)                                                                        \
    if (requested.name && !supportedCore.name)                                                     \
    unsupportedFeatures.push_back(#name)
    CHECK_FEATURE(robustBufferAccess);
    CHECK_FEATURE(fullDrawIndexUint32);
    CHECK_FEATURE(imageCubeArray);
    CHECK_FEATURE(independentBlend);
    CHECK_FEATURE(geometryShader);
    CHECK_FEATURE(tessellationShader);
    CHECK_FEATURE(sampleRateShading);
    CHECK_FEATURE(dualSrcBlend);
    CHECK_FEATURE(logicOp);
    CHECK_FEATURE(multiDrawIndirect);
    CHECK_FEATURE(drawIndirectFirstInstance);
    CHECK_FEATURE(depthClamp);
    CHECK_FEATURE(depthBiasClamp);
    CHECK_FEATURE(fillModeNonSolid);
    CHECK_FEATURE(depthBounds);
    CHECK_FEATURE(wideLines);
    CHECK_FEATURE(largePoints);
    CHECK_FEATURE(alphaToOne);
    CHECK_FEATURE(multiViewport);
    CHECK_FEATURE(samplerAnisotropy);
    CHECK_FEATURE(textureCompressionETC2);
    CHECK_FEATURE(textureCompressionASTC_LDR);
    CHECK_FEATURE(textureCompressionBC);
    CHECK_FEATURE(occlusionQueryPrecise);
    CHECK_FEATURE(pipelineStatisticsQuery);
    CHECK_FEATURE(vertexPipelineStoresAndAtomics);
    CHECK_FEATURE(fragmentStoresAndAtomics);
    CHECK_FEATURE(shaderTessellationAndGeometryPointSize);
    CHECK_FEATURE(shaderImageGatherExtended);
    CHECK_FEATURE(shaderStorageImageExtendedFormats);
    CHECK_FEATURE(shaderStorageImageMultisample);
    CHECK_FEATURE(shaderStorageImageReadWithoutFormat);
    CHECK_FEATURE(shaderStorageImageWriteWithoutFormat);
    CHECK_FEATURE(shaderUniformBufferArrayDynamicIndexing);
    CHECK_FEATURE(shaderSampledImageArrayDynamicIndexing);
    CHECK_FEATURE(shaderStorageBufferArrayDynamicIndexing);
    CHECK_FEATURE(shaderStorageImageArrayDynamicIndexing);
    CHECK_FEATURE(shaderClipDistance);
    CHECK_FEATURE(shaderCullDistance);
    CHECK_FEATURE(shaderFloat64);
    CHECK_FEATURE(shaderInt64);
    CHECK_FEATURE(shaderInt16);
    CHECK_FEATURE(shaderResourceResidency);
    CHECK_FEATURE(shaderResourceMinLod);
    CHECK_FEATURE(sparseBinding);
    CHECK_FEATURE(sparseResidencyBuffer);
    CHECK_FEATURE(sparseResidencyImage2D);
    CHECK_FEATURE(sparseResidencyImage3D);
    CHECK_FEATURE(sparseResidency2Samples);
    CHECK_FEATURE(sparseResidency4Samples);
    CHECK_FEATURE(sparseResidency8Samples);
    CHECK_FEATURE(sparseResidency16Samples);
    CHECK_FEATURE(sparseResidencyAliased);
    CHECK_FEATURE(variableMultisampleRate);
    CHECK_FEATURE(inheritedQueries);
#undef CHECK_FEATURE
    if (!unsupportedFeatures.empty())
    {
        std::string msg = "requested core features not supported:";
        for (auto &f : unsupportedFeatures)
            msg += " " + f;
        throw GpuError(msg);
    }

    // Feature structs to chain into device creation: those attached to an enabled
    // extension (skipped extensions drop theirs), plus the extension-less structs
    // requested via deviceFeatures. Bits not honored by the device fail creation.
    std::vector<const void *> toChain;
    for (const auto &ext : m_desc.deviceExtensions)
    {
        bool enabledExt = std::any_of(enabled.begin(), enabled.end(), [&ext](const char *e)
                                      { return ext.name && std::strcmp(e, ext.name) == 0; });
        if (enabledExt && ext.feature)
            toChain.push_back(ext.feature);
    }
    for (const auto *f : m_desc.deviceFeatures)
    {
        if (f)
            toChain.push_back(f);
    }

    // Build feature chain for device creation
    vk::PhysicalDeviceDescriptorIndexingFeatures enabledIndexingFeatures;
    enabledIndexingFeatures.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    enabledIndexingFeatures.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    enabledIndexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;

    vk::PhysicalDeviceTimelineSemaphoreFeatures enabledTimelineFeatures;
    enabledTimelineFeatures.timelineSemaphore = VK_TRUE;

    enabledIndexingFeatures.pNext = &enabledTimelineFeatures;
    enabledTimelineFeatures.pNext = linkFeatureChain(toChain);

    vk::DeviceCreateInfo devInfo({}, static_cast<uint32_t>(queueInfos.size()), queueInfos.data(), 0,
                                 nullptr, static_cast<uint32_t>(enabled.size()), enabled.data(),
                                 &m_desc.coreFeatures);
    devInfo.setPNext(&enabledIndexingFeatures);
    try
    {
        m_device = m_physicalDevice.createDeviceUnique(devInfo);
    }
    catch (const std::exception &e)
    {
        std::ostringstream os;
        os << "device creation failed (requested extensions:";
        for (const auto &ext : m_desc.deviceExtensions)
            os << " " << (ext.name ? ext.name : "<unnamed>");
        os << "): " << e.what();
        throw GpuError(os.str());
    }
    volkLoadDevice(*m_device);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_instance, vkGetInstanceProcAddr, *m_device,
                                       vkGetDeviceProcAddr);

    m_queue = m_device->getQueue(m_queueFamilyIndex, 0);
    m_transferQueue = m_device->getQueue(m_transferQueueFamilyIndex, 0);
}

void GpuContext::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                       m_queueFamilyIndex);
    m_commandPool = vkChecked([&] { return m_device->createCommandPoolUnique(poolInfo); },
                              "vkCreateCommandPool");

    if (m_hasSeparateTransferQueue && m_transferQueueFamilyIndex != m_queueFamilyIndex)
    {
        vk::CommandPoolCreateInfo transferPoolInfo(
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, m_transferQueueFamilyIndex);
        m_transferCommandPool =
            vkChecked([&] { return m_device->createCommandPoolUnique(transferPoolInfo); },
                      "vkCreateCommandPool (transfer)");
    }
    else
    {
        m_transferCommandPool =
            vkChecked([&] { return m_device->createCommandPoolUnique(poolInfo); },
                      "vkCreateCommandPool (transfer)");
    }
}

void GpuContext::createAllocator()
{
    VmaVulkanFunctions vmaFuncs = {};
    vmaFuncs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaFuncs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.vulkanApiVersion = m_desc.vulkanApiVersion;
    allocatorInfo.physicalDevice = m_physicalDevice;
    allocatorInfo.device = *m_device;
    allocatorInfo.instance = *m_instance;
    allocatorInfo.pVulkanFunctions = &vmaFuncs;
    vkCheck(vmaCreateAllocator(&allocatorInfo, &m_allocator), "vmaCreateAllocator");
}

void GpuContext::createPipelineCache()
{
    vk::PipelineCacheCreateInfo pcInfo;
    m_pipelineCache = vkChecked([&] { return m_device->createPipelineCacheUnique(pcInfo); },
                                "vkCreatePipelineCache");
    auto props = m_physicalDevice.getProperties();
    m_timestampPeriod = props.limits.timestampPeriod;
    m_deviceName = props.deviceName.data();
    m_limits = props.limits;
}

void GpuContext::setDebugName(vk::ObjectType type, uint64_t object, const std::string &name)
{
    if (!m_debugUtilsEnabled)
        return;
    vk::DebugUtilsObjectNameInfoEXT info;
    info.setObjectType(type);
    info.setObjectHandle(object);
    info.setPObjectName(name.c_str());
    m_device->setDebugUtilsObjectNameEXT(info);
}

void GpuContext::setDebugName(vk::Buffer object, const std::string &name)
{
    setDebugName(vk::ObjectType::eBuffer, reinterpret_cast<uint64_t>(static_cast<VkBuffer>(object)),
                 name);
}

void GpuContext::setDebugName(vk::Image object, const std::string &name)
{
    setDebugName(vk::ObjectType::eImage, reinterpret_cast<uint64_t>(static_cast<VkImage>(object)),
                 name);
}

void GpuContext::setDebugName(vk::Pipeline object, const std::string &name)
{
    setDebugName(vk::ObjectType::ePipeline,
                 reinterpret_cast<uint64_t>(static_cast<VkPipeline>(object)), name);
}

void GpuContext::setDebugName(vk::Semaphore object, const std::string &name)
{
    setDebugName(vk::ObjectType::eSemaphore,
                 reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(object)), name);
}

void GpuContext::setDebugName(vk::CommandBuffer object, const std::string &name)
{
    setDebugName(vk::ObjectType::eCommandBuffer,
                 reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(object)), name);
}

void GpuContext::setDebugName(vk::DescriptorSet object, const std::string &name)
{
    setDebugName(vk::ObjectType::eDescriptorSet,
                 reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(object)), name);
}

bool GpuContext::hasDeviceExtension(const char *name) const noexcept
{
    if (!name)
        return false;
    return std::any_of(m_deviceExtensions.begin(), m_deviceExtensions.end(),
                       [name](const std::string &e) { return std::strcmp(e.c_str(), name) == 0; });
}

MemoryStats GpuContext::memoryStats() const
{
    auto memProps = m_physicalDevice.getMemoryProperties();
    uint32_t heapCount = memProps.memoryHeapCount;

    std::vector<VmaBudget> budgets(heapCount);
    if (heapCount > 0)
        vmaGetHeapBudgets(m_allocator, budgets.data());

    MemoryStats stats;
    stats.heaps.reserve(heapCount);
    for (uint32_t i = 0; i < heapCount; ++i)
    {
        MemoryHeapStats heap;
        heap.heapIndex = i;
        heap.deviceLocal = (memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) !=
                           vk::MemoryHeapFlags{};
        heap.budget = budgets[i].budget;
        heap.usage = budgets[i].usage;
        heap.blockBytes = budgets[i].statistics.blockBytes;
        heap.allocationBytes = budgets[i].statistics.allocationBytes;
        stats.heaps.push_back(heap);
        stats.totalBudget += budgets[i].budget;
        stats.totalUsage += budgets[i].usage;
    }
    return stats;
}

void GpuContext::reportGpuFailure(vk::Result result, const std::string &operation,
                                  const std::vector<std::string> &recentLabels)
{
    m_lastCrashInfo.valid = true;
    m_lastCrashInfo.result = result;
    m_lastCrashInfo.message =
        operation + " (VkResult " + std::to_string(static_cast<int>(result)) + ")";
    m_lastCrashInfo.deviceName = m_deviceName;
    m_lastCrashInfo.recentLabels = recentLabels;
    try
    {
        m_lastCrashInfo.infoDump = infoDump();
    }
    catch (...)
    {
        m_lastCrashInfo.infoDump.clear();
    }

    if (!m_deviceFaultEnabled || !m_device)
        return;
    try
    {
        vk::DeviceFaultCountsEXT counts;
        if (m_device->getFaultInfoEXT(&counts, nullptr) != vk::Result::eSuccess)
            return;
        std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
        std::vector<vk::DeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
        std::vector<uint8_t> binary(static_cast<size_t>(counts.vendorBinarySize));
        vk::DeviceFaultInfoEXT details;
        details.pAddressInfos = addresses.data();
        details.pVendorInfos = vendors.data();
        details.pVendorBinaryData = binary.data();
        if (m_device->getFaultInfoEXT(&counts, &details) == vk::Result::eSuccess)
            m_lastCrashInfo.deviceFault = details.description.data();
    }
    catch (...)
    {
        m_lastCrashInfo.deviceFault.clear();
    }
}

std::string GpuContext::infoDump() const
{
    std::ostringstream os;
    auto props = m_physicalDevice.getProperties();

    os << "pixie_compute info dump\n";
    os << "  device:              " << props.deviceName.data() << "\n";
    os << "  deviceType:          " << vk::to_string(props.deviceType) << "\n";
    os << "  instanceApiVersion:  " << VK_API_VERSION_MAJOR(m_desc.vulkanApiVersion) << "."
       << VK_API_VERSION_MINOR(m_desc.vulkanApiVersion) << "."
       << VK_API_VERSION_PATCH(m_desc.vulkanApiVersion) << "\n";
    os << "  apiVersion:          " << VK_API_VERSION_MAJOR(props.apiVersion) << "."
       << VK_API_VERSION_MINOR(props.apiVersion) << "." << VK_API_VERSION_PATCH(props.apiVersion)
       << "\n";
    os << "  validation:          " << (m_validation ? "enabled" : "disabled") << "\n";
    os << "  shaderPrintf:        " << (m_shaderPrintfEnabled ? "enabled" : "disabled") << "\n";
    os << "  driverVersion:       " << props.driverVersion << " (vendor-encoded)\n";

    os << "\n  queueFamilies:\n";
    auto queueProps = m_physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i)
    {
        os << "    [" << i << "] flags=" << vk::to_string(queueProps[i].queueFlags)
           << " count=" << queueProps[i].queueCount << (i == m_queueFamilyIndex ? " (compute)" : "")
           << (m_hasSeparateTransferQueue && i == m_transferQueueFamilyIndex ? " (transfer)" : "")
           << "\n";
    }

    os << "\n  instanceExtensions:\n";
    if (m_instanceExtensions.empty())
        os << "    (none)\n";
    for (const auto &ext : m_instanceExtensions)
        os << "    " << ext << "\n";

    os << "\n  deviceExtensions:\n";
    if (m_deviceExtensions.empty())
        os << "    (none)\n";
    for (const auto &ext : m_deviceExtensions)
        os << "    " << ext << "\n";

    if (!m_skippedDeviceExtensions.empty())
    {
        os << "\n  requested but unavailable:\n";
        for (const auto &ext : m_skippedDeviceExtensions)
            os << "    " << ext << "\n";
    }

    os << "\n  memoryHeaps:\n";
    auto memProps = m_physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
    {
        os << "    [" << i << "] size=" << formatBytes(memProps.memoryHeaps[i].size) << " flags=";
        if (memProps.memoryHeaps[i].flags == vk::MemoryHeapFlagBits::eDeviceLocal)
            os << "DeviceLocal";
        else if (memProps.memoryHeaps[i].flags == vk::MemoryHeapFlags{})
            os << "host";
        else
            os << vk::to_string(memProps.memoryHeaps[i].flags);
        os << "\n";
    }

    os << "\n  limits:\n";
    os << "    maxComputeWorkGroupCount=( " << props.limits.maxComputeWorkGroupCount[0] << ", "
       << props.limits.maxComputeWorkGroupCount[1] << ", "
       << props.limits.maxComputeWorkGroupCount[2] << " )\n";
    os << "    timestampComputeAndGraphics="
       << (props.limits.timestampComputeAndGraphics ? "true" : "false") << "\n";

    auto stats = memoryStats();
    os << "\n  memoryStats:\n";
    for (const auto &heap : stats.heaps)
    {
        os << "    heap[" << heap.heapIndex << "] " << (heap.deviceLocal ? "deviceLocal " : "host ")
           << "budget=" << formatBytes(heap.budget) << " usage=" << formatBytes(heap.usage)
           << " blockBytes=" << formatBytes(heap.blockBytes)
           << " allocationBytes=" << formatBytes(heap.allocationBytes) << "\n";
    }
    os << "    totalBudget=" << formatBytes(stats.totalBudget)
       << " totalUsage=" << formatBytes(stats.totalUsage) << "\n";
    return os.str();
}

std::vector<uint8_t> GpuContext::savePipelineCache() const
{
    auto data = m_device->getPipelineCacheData(*m_pipelineCache);
    return std::vector<uint8_t>(data.begin(), data.end());
}
void GpuContext::loadPipelineCache(const std::vector<uint8_t> &data)
{
    vk::PipelineCacheCreateInfo pcInfo;
    pcInfo.initialDataSize = data.size();
    pcInfo.pInitialData = data.data();
    m_pipelineCache = vkChecked([&] { return m_device->createPipelineCacheUnique(pcInfo); },
                                "vkCreatePipelineCache");
}

void GpuContext::waitForIdle()
{
    m_device->waitIdle();
}

StagingBuffer GpuContext::acquireStagingBuffer(size_t minSize)
{
    std::lock_guard lock(m_stagingMutex);

    // Search for a pooled buffer that fits
    for (auto it = m_stagingPool.begin(); it != m_stagingPool.end(); ++it)
    {
        if (it->size >= minSize)
        {
            StagingBuffer buf = std::move(*it);
            *it = std::move(m_stagingPool.back());
            m_stagingPool.pop_back();
            return buf;
        }
    }

    // No suitable buffer in pool — allocate a new one
    vk::BufferCreateInfo bufferInfo(
        {}, minSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo resultInfo = {};
    StagingBuffer buf{};
    auto result =
        vmaCreateBuffer(m_allocator, reinterpret_cast<const VkBufferCreateInfo *>(&bufferInfo),
                        &allocInfo, &buf.buffer, &buf.allocation, &resultInfo);
    if (result != VK_SUCCESS)
        throw GpuError("VMA staging buffer allocation failed (VkResult " +
                       std::to_string(static_cast<int>(result)) + ")");
    const VkPhysicalDeviceMemoryProperties *memProps = nullptr;
    vmaGetMemoryProperties(m_allocator, &memProps);
    const VkMemoryPropertyFlags props = memProps->memoryTypes[resultInfo.memoryType].propertyFlags;
    if ((props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
    {
        vmaDestroyBuffer(m_allocator, buf.buffer, buf.allocation);
        throw GpuError("staging buffer memory is not host-coherent on this device; "
                       "uploads/downloads would require explicit cache management");
    }
    buf.mapped = resultInfo.pMappedData;
    buf.size = minSize;
    return buf;
}

void GpuContext::releaseStagingBuffer(StagingBuffer buf)
{
    if (!buf.buffer)
        return;
    std::lock_guard lock(m_stagingMutex);
    m_stagingPool.push_back(buf);
}

GpuTimelineSemaphore GpuContext::createTimelineSemaphore(uint64_t initialValue)
{
    return GpuTimelineSemaphore(*this, initialValue);
}

GpuContext::~GpuContext()
{
    if (m_device)
        m_device->waitIdle();

    // Destroy pooled staging buffers
    for (auto &buf : m_stagingPool)
    {
        if (buf.buffer && m_allocator)
            vmaDestroyBuffer(m_allocator, buf.buffer, buf.allocation);
    }
    m_stagingPool.clear();

    if (m_allocator)
    {
        vmaDestroyAllocator(m_allocator);
    }
}

} // namespace pix
