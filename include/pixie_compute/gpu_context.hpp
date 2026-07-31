#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>
#include <vulkan/vulkan.hpp>

struct VmaAllocator_T;

namespace pix
{

struct StagingBuffer;
class GpuTimelineSemaphore;

/// VMA usage/budget statistics for a single memory heap.
struct MemoryHeapStats
{
    uint32_t heapIndex = 0;
    bool deviceLocal = false;
    /// Size of the heap reported by the driver (may be 0 if unknown).
    VkDeviceSize budget = 0;
    /// Current bytes in use on the heap (usage only, not reservations).
    VkDeviceSize usage = 0;
    /// Bytes reserved by VMA blocks on this heap.
    VkDeviceSize blockBytes = 0;
    /// Bytes allocated from VMA blocks (sub-allocations).
    VkDeviceSize allocationBytes = 0;
};

/// Per-heap VMA budget statistics plus aggregated totals.
struct MemoryStats
{
    std::vector<MemoryHeapStats> heaps;
    VkDeviceSize totalBudget = 0;
    VkDeviceSize totalUsage = 0;
};

/// A device extension to enable, together with optional behavior when the
/// device does not support it.
struct GpuExtension
{
    /// Extension name, e.g. VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME.
    const char *name = nullptr;
    /// Called when the device does not support the extension. Returns true to
    /// continue without it (the extension is skipped and reported via infoDump());
    /// false, or a null callback, aborts context creation with a GpuError.
    std::function<bool(const char *)> onMissing;
    /// Optional feature struct whose bits are set to VK_TRUE (e.g.
    /// VkPhysicalDeviceCooperativeMatrixFeaturesKHR). The struct is chained into
    /// device creation only when the extension is enabled; when the extension is
    /// missing or skipped, the feature is silently dropped as well. Must outlive
    /// the GpuContext.
    const void *feature = nullptr;
    /// Required-extension shorthand: GpuExtension{"VK_KHR_FOO"} behaves like a
    /// null onMissing callback (missing extension throws).
    GpuExtension(const char *n) : name(n) {}
    GpuExtension(const char *n, std::function<bool(const char *)> cb) : name(n), onMissing(std::move(cb)) {}
    GpuExtension(const char *n, std::function<bool(const char *)> cb, const void *f)
        : name(n), onMissing(std::move(cb)), feature(f)
    {
    }
};

/// Configuration for GpuContext initialization.
struct GpuContextDesc
{
    /// Vulkan API version for the instance and device. Must be at least 1.1.
    /// When 1.2 or newer, the descriptor indexing and timeline semaphore features
    /// are provided as core functionality, so their extensions are not requested.
    uint32_t vulkanApiVersion = VK_API_VERSION_1_1;
    /// Additional Vulkan instance extensions to enable (e.g. VK_EXT_DEBUG_UTILS_EXTENSION_NAME).
    std::vector<std::string> instanceExtensions;
    /// Additional device extensions to enable. Each entry may carry an onMissing
    /// callback to allow skipping; without one, a missing extension throws.
    std::vector<GpuExtension> deviceExtensions;
    /// Feature structs with no extension gate (e.g. core features such as
    /// shaderFloat16), always requested. Unsupported bits make device creation fail.
    /// Each struct must outlive the GpuContext.
    std::vector<const void *> deviceFeatures;
    /// Core Vulkan features to enable (VkPhysicalDeviceFeatures). Default: all off.
    /// Only features supported by the device are honored; unsupported bits cause
    /// device creation to fail with a descriptive GpuError.
    vk::PhysicalDeviceFeatures coreFeatures{};
    /// Optional VkSurfaceKHR for presentation support. Pass VK_NULL_HANDLE for headless compute.
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    /// If true, the selected queue family will also support graphics operations.
    bool requireGraphics = false;
    /// Optional physical-device selector. When set, the device with the highest
    /// score wins (ties: first enumerated). When empty, a built-in heuristic
    /// prefers discrete GPUs, then largest maxComputeWorkGroupCount[0].
    std::function<uint32_t(vk::PhysicalDevice, const vk::PhysicalDeviceProperties &)> deviceScore;
    /// Enable the Vulkan validation layers and a debug-utils messenger.
    /// Defaults to on for debug builds and off for release builds. May be overridden at
    /// runtime with the PIXIE_COMPUTE_VALIDATION environment variable ("0"/"1").
#ifdef NDEBUG
    bool enableValidation = false;
#else
    bool enableValidation = true;
#endif
    /// Severity filter for validation messages delivered to the debug messenger.
    vk::DebugUtilsMessageSeverityFlagsEXT debugSeverity =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    /// Optional sink for validation messages. When empty, messages are printed to stderr
    /// (Android: logcat).
    std::function<void(vk::DebugUtilsMessageSeverityFlagBitsEXT, const std::string &)> logSink;
    /// If true, raise a breakpoint (SIGTRAP, or __debugbreak on Windows) when a validation
    /// error is delivered to the debug messenger. Only effective when validation is enabled.
    /// Intended for use with a debugger attached; without one, this terminates the process.
    bool breakOnError = false;
};

/// Manages Vulkan instance, device, queue, command pool, and VMA allocator.
///
/// Thread-safety: volk and vulkan.hpp use process-global dispatch tables. Multiple
/// GpuContext instances on *different* devices clobber each other's device-level
/// dispatch. One context per process is the supported configuration.
///
/// Destruction order: GpuContext must outlive all resources created from it
/// (GpuBuffer, GpuImage, GpuComputePipeline, GpuCommandBuffer, GpuTimelineSemaphore,
/// GpuTimer, GpuProfiler). The context destructor calls device.waitIdle() before
/// cleanup.
class GpuContext
{
    friend class GpuBuffer;
    friend class GpuCommandBuffer;
    friend class GpuImage;

    public:
    explicit GpuContext(const GpuContextDesc &desc = {});
    ~GpuContext();

    GpuContext(const GpuContext &) = delete;
    GpuContext &operator=(const GpuContext &) = delete;
    GpuContext(GpuContext &&) = delete;
    GpuContext &operator=(GpuContext &&) = delete;

    vk::Instance instance() const noexcept { return *m_instance; }
    vk::Device device() const noexcept { return *m_device; }
    vk::PhysicalDevice physicalDevice() const noexcept { return m_physicalDevice; }
    vk::Queue queue() const noexcept { return m_queue; }
    uint32_t queueFamilyIndex() const noexcept { return m_queueFamilyIndex; }
    vk::CommandPool commandPool() const noexcept { return *m_commandPool; }
    VmaAllocator_T *allocator() const noexcept { return m_allocator; }
    vk::PipelineCache pipelineCache() const noexcept { return *m_pipelineCache; }
    float timestampPeriod() const noexcept { return m_timestampPeriod; }
    const std::string &deviceName() const noexcept { return m_deviceName; }
    const vk::PhysicalDeviceLimits &limits() const noexcept { return m_limits; }
    std::vector<uint8_t> savePipelineCache() const;
    void loadPipelineCache(const std::vector<uint8_t> &data);
    void waitForIdle();

    bool hasSeparateTransferQueue() const noexcept { return m_hasSeparateTransferQueue; }
    vk::Queue transferQueue() const noexcept { return m_transferQueue; }
    vk::CommandPool transferCommandPool() const noexcept { return *m_transferCommandPool; }
    /// True if the validation layers and debug messenger are active for this context.
    bool validationEnabled() const noexcept { return m_validation; }
    /// True if VK_EXT_debug_utils is enabled, so object names and command labels take effect.
    bool debugUtilsEnabled() const noexcept { return m_debugUtilsEnabled; }
    /// True if the named device extension is enabled on this context (includes
    /// the extensions the library enables internally).
    bool hasDeviceExtension(const char *name) const noexcept;

    /// Assign a debug name to a Vulkan object (VK_EXT_debug_utils). No-op when the
    /// extension is unavailable. Names show up in the validation layer, RenderDoc, and
    /// graphics debuggers.
    void setDebugName(vk::Buffer object, const std::string &name);
    void setDebugName(vk::Image object, const std::string &name);
    void setDebugName(vk::Pipeline object, const std::string &name);
    void setDebugName(vk::Semaphore object, const std::string &name);
    void setDebugName(vk::CommandBuffer object, const std::string &name);
    void setDebugName(vk::DescriptorSet object, const std::string &name);

    /// VMA memory usage/budget per heap, plus aggregates. Useful for diagnostics and
    /// memory-leak checks (usage grows as allocations are made and not freed).
    MemoryStats memoryStats() const;

    /// Multi-line diagnostic string with device/queue/extension/feature/heap summary.
    std::string infoDump() const;

    /// Create a timeline semaphore starting at initialValue.
    GpuTimelineSemaphore createTimelineSemaphore(uint64_t initialValue = 0);

    private:
    void selectPhysicalDevice();
    void findQueueFamily();
    void findTransferQueueFamily();
    void createDevice();
    void createCommandPool();
    void createAllocator();
    void createPipelineCache();
    void setDebugName(vk::ObjectType type, uint64_t object, const std::string &name);

    StagingBuffer acquireStagingBuffer(size_t minSize);
    void releaseStagingBuffer(StagingBuffer buf);

    std::mutex &queueMutex() noexcept { return m_queueMutex; }
    std::mutex &transferQueueMutex() noexcept { return m_transferQueueMutex; }

    GpuContextDesc m_desc;
    vk::UniqueInstance m_instance;
    vk::UniqueDebugUtilsMessengerEXT m_debugMessenger;
    bool m_validation = false;
    bool m_debugUtilsEnabled = false;
    vk::PhysicalDevice m_physicalDevice;
    vk::UniqueDevice m_device;
    uint32_t m_queueFamilyIndex = 0;
    vk::Queue m_queue;
    vk::UniqueCommandPool m_commandPool;
    uint32_t m_transferQueueFamilyIndex = 0;
    vk::Queue m_transferQueue;
    vk::UniqueCommandPool m_transferCommandPool;
    bool m_hasSeparateTransferQueue = false;
    VmaAllocator_T *m_allocator = nullptr;
    vk::UniquePipelineCache m_pipelineCache;
    float m_timestampPeriod = 1.0f;
    std::string m_deviceName;
    vk::PhysicalDeviceLimits m_limits{};
    std::vector<std::string> m_instanceExtensions;
    std::vector<std::string> m_deviceExtensions;
    std::vector<std::string> m_skippedDeviceExtensions;
    std::mutex m_queueMutex;
    std::mutex m_transferQueueMutex;
    std::vector<StagingBuffer> m_stagingPool;
    std::mutex m_stagingMutex;
};

} // namespace pix
