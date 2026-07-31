#pragma once

#include "pixie_compute/detail/vulkan_include.hpp"
#include "pixie_compute/utility.hpp"

namespace pix
{
namespace detail
{

/// Read a begin/end timestamp pair from a query pool and convert to
/// milliseconds. Blocks until the results are available.
inline double readTimestampMs(vk::Device device, vk::QueryPool pool, uint32_t pairIndex,
                              float timestampPeriod)
{
    uint64_t timestamps[2]{};
    auto result = device.getQueryPoolResults(
        pool, pairIndex * 2, 2, sizeof(timestamps), timestamps, sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (result != vk::Result::eSuccess)
        throw GpuError("getQueryPoolResults failed (VkResult " +
                       std::to_string(static_cast<int>(result)) + ")");
    double ns = static_cast<double>(timestamps[1] - timestamps[0]) * timestampPeriod;
    return ns / 1.0e6;
}

} // namespace detail
} // namespace pix
