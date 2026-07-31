#pragma once

#include "pixie_compute/detail/vulkan_include.hpp"
#include "pixie_compute/utility.hpp"

#include <string>

namespace pix
{

/// Run a vulkan.hpp call, converting any vk::Error into GpuError with context.
template <typename F> decltype(auto) vkChecked(F &&f, const char *what)
{
    try
    {
        return f();
    }
    catch (const vk::Error &e)
    {
        throw GpuError(std::string(what) + ": " + e.what());
    }
}

/// Check a raw VkResult-returning call (VMA, C API).
inline void vkCheck(VkResult result, const char *what)
{
    if (result != VK_SUCCESS)
        throw GpuError(std::string(what) + " (VkResult " +
                       std::to_string(static_cast<int>(result)) + ")");
}

} // namespace pix
