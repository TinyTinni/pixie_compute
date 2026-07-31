#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_timeline_semaphore.hpp"
#include "pixie_compute/utility.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace pix;

namespace
{

static constexpr const char *kFakeExtension = "VK_KHR_FAKE_NONEXISTENT_EXTENSION";

static bool continueWithout(const char *name)
{
    REQUIRE(name);
    return true;
}

static bool abortWithout(const char *name)
{
    REQUIRE(name);
    return false;
}

} // namespace

TEST_CASE("required extension missing throws GpuError", "[device_extensions]")
{
    GpuContextDesc desc;
    desc.deviceExtensions.push_back(kFakeExtension);
    REQUIRE_THROWS_AS(GpuContext(desc), GpuError);
}

TEST_CASE("onMissing returning false throws GpuError", "[device_extensions]")
{
    GpuContextDesc desc;
    desc.deviceExtensions.push_back({kFakeExtension, &abortWithout});
    REQUIRE_THROWS_AS(GpuContext(desc), GpuError);
}

TEST_CASE("onMissing returning true skips the extension and reports it", "[device_extensions]")
{
    GpuContextDesc desc;
    desc.deviceExtensions.push_back({VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, &continueWithout});
    GpuContext ctx(desc);

    if (!ctx.hasDeviceExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
    {
        auto dump = ctx.infoDump();
        REQUIRE(dump.find("requested but unavailable") != std::string::npos);
        REQUIRE(dump.find(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) != std::string::npos);
    }
}

TEST_CASE("hasDeviceExtension reflects enabled extensions", "[device_extensions]")
{
    GpuContextDesc desc;
    desc.deviceExtensions.push_back({VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, &continueWithout});
    GpuContext ctx(desc);

    // Internal required extension is always enabled.
    REQUIRE(ctx.hasDeviceExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME));
    // Fake extension never is.
    REQUIRE_FALSE(ctx.hasDeviceExtension(kFakeExtension));
}

TEST_CASE("vulkanApiVersion 1.3 drops core-1.2 extensions but keeps features", "[device_extensions]")
{
    GpuContextDesc desc;
    desc.vulkanApiVersion = VK_API_VERSION_1_3;
    GpuContext ctx(desc);

    // Descriptor indexing and timeline semaphores are core since 1.2 and must not
    // be requested as extensions.
    REQUIRE_FALSE(ctx.hasDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME));
    REQUIRE_FALSE(ctx.hasDeviceExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME));

    // Core timeline-semaphore functionality still works: an empty submission signals
    // the semaphore, and a second submission waits on it.
    GpuTimelineSemaphore sem(ctx);
    uint64_t val = sem.next();

    GpuCommandBuffer signalCmd(ctx);
    signalCmd.begin();
    signalCmd.end();
    signalCmd.submit({.signals = {GpuSignalSemaphore::makeTimeline(sem, val)}});

    GpuCommandBuffer waitCmd(ctx);
    waitCmd.begin();
    waitCmd.end();
    REQUIRE_NOTHROW(waitCmd.submitAndWait({.waits = {GpuWaitSemaphore::makeTimeline(sem, val)}}));
    REQUIRE(sem.currentValue() >= val);
}
