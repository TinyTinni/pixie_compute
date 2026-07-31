#pragma once

#include "pixie_compute/gpu_context.hpp"

#include <catch2/catch_test_macros.hpp>
#include <memory>

namespace pix
{

/// Shared test fixture that provides a single GpuContext for all tests.
/// Skips the test if Vulkan is not available.
struct GpuTestFixture
{
    static GpuContext &ctx()
    {
        static auto instance = []() -> std::unique_ptr<GpuContext>
        {
            try
            {
                return std::make_unique<GpuContext>();
            }
            catch (...)
            {
                return nullptr;
            }
        }();
        if (!instance)
            SKIP("Vulkan not available");
        return *instance;
    }
};

} // namespace pix
