// ============================================================================
// pixie_compute template — start here to implement your GPU compute idea.
//
// Layout:
//   src/main.cpp           the host program (this file)
//   src/vector_add.{hpp,cpp}  a reusable compute function shared with the tests
//   shaders/vector_add.slang  the kernel, compiled at runtime to SPIR-V
//   tests/                  Catch2 tests (see tests/test_vector_add.cpp)
//
// Mental model (everything in the library is RAII — no manual cleanup):
//
//   1. GpuContext       — one per process. Creates the Vulkan instance,
//                         device, queues, command pools, and memory allocator.
//                         Headless by default: no window needed.
//   2. GpuBuffer        — GPU memory. HostCoherent buffers are CPU-visible and
//                         the simplest to work with; Device buffers are faster
//                         for the GPU but copies go through staging.
//   3. Shader + pipeline— a compute kernel written in Slang (an HLSL-like
//                         language). compileSlangModule() turns the .slang file
//                         into SPIR-V; GpuComputePipeline builds the Vulkan
//                         pipeline from it.
//   4. Dispatch         — oneShotDispatch() records, submits, and waits in a
//                         single call. For multiple passes use GpuCommandBuffer.
//   5. Read back        — download() copies results back to a std::vector.
// ============================================================================

#include "pixie_compute/pixie_compute.hpp"
#include "vector_add.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace pix;

int main()
{
    try
    {
        // --- STEP 1: create the context -----------------------------------
        GpuContext ctx;
        std::printf("[template] GPU: %s\n", ctx.deviceName().c_str());
        // Validation layers are enabled automatically in debug builds, and the
        // device/queue/extension summary is available for troubleshooting:
        //     std::printf("%s\n", ctx.infoDump().c_str());

        // --- STEP 2: prepare data -----------------------------------------
        constexpr uint32_t N = 1 << 20; // 1M elements

        std::vector<float> hostA(N), hostB(N);
        for (uint32_t i = 0; i < N; ++i)
        {
            hostA[i] = static_cast<float>(i);
            hostB[i] = static_cast<float>(i) * 0.5f;
        }

        // --- STEP 3+4+5: compute on the GPU --------------------------------
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> hostC = vectorAdd(ctx, hostA, hostB); // dispatch + read back
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- verify --------------------------------------------------------
        float maxErr = 0.0f;
        for (uint32_t i = 0; i < N; ++i)
            maxErr = std::fmax(maxErr, std::fabs(hostC[i] - (hostA[i] + hostB[i])));

        std::printf("[template] N=%u  compute took %.3f ms  max error: %.3e  %s\n",
                    N, ms, maxErr, maxErr < 1e-4f ? "PASS" : "FAIL");
        return maxErr < 1e-4f ? 0 : 1;

        // ====================================================================
        // ---- YOUR IDEA GOES HERE -------------------------------------------
        // Replace the shader in shaders/vector_add.slang, adjust N above, and
        // repeat. The plumbing stays. For multiple passes, push constants, or
        // async work, use a command buffer instead of oneShotDispatch():
        //
        //     GpuCommandBuffer cmd(ctx);
        //     cmd.begin();
        //     cmd.bind(pipeline);
        //     cmd.pushConstants(&myConstants, sizeof(myConstants));
        //     cmd.dispatch(groupsX, groupsY, groupsZ);
        //     cmd.end();
        //     cmd.submitAndWait();
        //
        // Push constants are declared in the shader as
        // "[[vk::push_constant]] cbuffer PC { ... };"
        // ====================================================================
    }
    catch (const GpuError &e)
    {
        // All failures — Vulkan init, shader compile, dispatch, allocation —
        // surface as pix::GpuError (or a std::exception).
        std::fprintf(stderr, "[template] ERROR: %s\n", e.what());
        return 1;
    }
}
