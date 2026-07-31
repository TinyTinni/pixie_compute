#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pix;

namespace
{

VkDeviceSize baselineTotalAllocation(const MemoryStats &stats)
{
    VkDeviceSize total = 0;
    for (const auto &heap : stats.heaps)
        total += heap.allocationBytes;
    return total;
}

} // namespace

TEST_CASE("memoryStats has per-heap entries with consistent aggregates", "[memory]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto stats = ctx.memoryStats();

    REQUIRE_FALSE(stats.heaps.empty());

    VkDeviceSize budgetSum = 0;
    VkDeviceSize usageSum = 0;
    for (const auto &heap : stats.heaps)
    {
        budgetSum += heap.budget;
        usageSum += heap.usage;
    }
    REQUIRE(stats.totalBudget == budgetSum);
    REQUIRE(stats.totalUsage == usageSum);
}

TEST_CASE("memoryStats reflects a buffer allocation", "[memory]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto baseline = ctx.memoryStats();

    {
        GpuBuffer buf(ctx, 16 * 1024 * 1024, GpuBuffer::Type::Device);
        auto stats = ctx.memoryStats();
        VkDeviceSize totalAlloc = 0;
        for (const auto &heap : stats.heaps)
            totalAlloc += heap.allocationBytes;
        REQUIRE(totalAlloc > baselineTotalAllocation(baseline));
    }
}

TEST_CASE("memoryStats after freeing returns toward baseline", "[memory]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto baseline = ctx.memoryStats();

    {
        GpuBuffer buf(ctx, 16 * 1024 * 1024, GpuBuffer::Type::Device);
        auto during = ctx.memoryStats();
        REQUIRE(baselineTotalAllocation(during) > baselineTotalAllocation(baseline));
    }

    auto after = ctx.memoryStats();
    REQUIRE(baselineTotalAllocation(after) <= baselineTotalAllocation(baseline));
}
