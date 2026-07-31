#pragma once

#include "pixie_compute/utility.hpp"

#include <string>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <volk.h>
#include <vulkan/vulkan.hpp>

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
        throw GpuError(std::string(what) + " (VkResult " + std::to_string(static_cast<int>(result)) +
                       ")");
}

} // namespace pix
