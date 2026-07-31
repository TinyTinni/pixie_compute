#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

using namespace pix;

// ---------------------------------------------------------------------------
// Cooperative-matrix example with a device-query-driven kernel.
//
// The kernel is a single static Slang shader parameterized by compile-time
// macros (TILE_M / TILE_N / TILE_K / SUBGROUP_SIZE), so no shader text is ever
// generated. Before compiling, the example queries the physical device for its
// supported VkCooperativeMatrixPropertiesKHR and picks a configuration whose
// tile shape actually matches the hardware:
//
//   * AType/BType = SINT8, CType/ResultType = SINT32, scope = SUBGROUP,
//     non-saturating int32 accumulation,
//   * square tiles (M == N) that divide the problem size,
//   * the largest such tile (fewest workgroups, best efficiency).
//
// The chosen values are fed into the shader via ShaderOptions::macros and the
// shader is compiled targeting SPIR-V 1.6, which the CooperativeMatrixKHR
// capability requires. One workgroup (one subgroup) computes one output tile
// derived from SV_GroupID, so all threads in the subgroup agree on tile
// coordinates.
//
// Requires VK_KHR_cooperative_matrix and Vulkan 1.3 (SPIR-V 1.6, which the
// CooperativeMatrixKHR capability requires, is only available from 1.3 onward).
// When the device lacks the extension or has no matching configuration, the
// example prints why (infoDump) and exits 0.
// ---------------------------------------------------------------------------

// Called when the device does not support VK_KHR_cooperative_matrix.
// Returning true lets context creation continue without the extension.
static bool continueWithoutCoopMatrix(const char *name)
{
    std::printf("[cooperative_matrix] %s not supported; continuing without it\n", name);
    return true;
}

// Problem size for the matmul (must be divisible by every candidate tile).
static constexpr uint32_t kProblemM = 32, kProblemN = 32, kProblemK = 32;

// Single static kernel. TILE_M / TILE_N / TILE_K / SUBGROUP_SIZE are normally
// supplied as compile-time macros by the host; the #ifndef guards keep the
// source self-contained (used by the unit test with the defaults).
static constexpr const char *kShader = R"(
using namespace linalg;

#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef TILE_K
#define TILE_K 16
#endif
#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

[[vk::binding(0, 0)]] RWStructuredBuffer<int8_t> a;
[[vk::binding(1, 0)]] RWStructuredBuffer<int8_t> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<int32_t> c;

[[vk::push_constant]] cbuffer PC { uint4 params; };

[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID)
{
    uint M = params.x;
    uint N = params.y;
    uint K = params.z;

    uint tileM = gid.y; // tile row of C
    uint tileN = gid.x; // tile col of C

    CoopMat<int32_t, MemoryScope.Subgroup, TILE_M, TILE_N, CoopMatMatrixUse::MatrixAccumulator> acc;
    acc.fill(int32_t(0));

    for (uint k = 0; k < K / TILE_K; ++k)
    {
        let aElem = tileM * TILE_M * K + k * TILE_K;
        let bElem = k * TILE_K * N + tileN * TILE_N;

        let aTile = coopMatLoad<int8_t, MemoryScope.Subgroup, TILE_M, TILE_K, CoopMatMatrixUse::MatrixA, CoopMatMatrixLayout::RowMajor>(a, aElem, K);
        let bTile = coopMatLoad<int8_t, MemoryScope.Subgroup, TILE_K, TILE_N, CoopMatMatrixUse::MatrixB, CoopMatMatrixLayout::RowMajor>(b, bElem, N);

        acc = coopMatMulAdd<int32_t, false>(aTile, bTile, acc);
    }

    acc.Store<CoopMatMatrixLayout::RowMajor>(c, tileM * TILE_M * N + tileN * TILE_N, N);
}
)";

struct CoopMatrixConfig
{
    uint32_t tileM = 0, tileN = 0, tileK = 0;
    uint32_t subgroupSize = 0;
    bool found = false;
};

// Query the device for a cooperative-matrix configuration this example can
// use, preferring the largest supported tile.
static CoopMatrixConfig findCoopMatrixConfig(const GpuContext &ctx)
{
    VkPhysicalDeviceSubgroupProperties subgroupProps = {};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(ctx.physicalDevice(), &props2);

    uint32_t propCount = 0;
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(ctx.physicalDevice(), &propCount, nullptr);
    std::vector<VkCooperativeMatrixPropertiesKHR> props(propCount);
    if (propCount > 0)
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(ctx.physicalDevice(), &propCount,
                                                          props.data());

    std::printf("[cooperative_matrix] device supports %u cooperative matrix type(s), "
                "subgroup size %u:\n",
                propCount, subgroupProps.subgroupSize);
    for (const auto &p : props)
    {
        std::printf("    M=%u N=%u K=%u AType=0x%x BType=0x%x CType=0x%x ResultType=0x%x "
                    "scope=0x%x saturating=%u\n",
                    p.MSize, p.NSize, p.KSize, static_cast<uint32_t>(p.AType),
                    static_cast<uint32_t>(p.BType), static_cast<uint32_t>(p.CType),
                    static_cast<uint32_t>(p.ResultType), static_cast<uint32_t>(p.scope),
                    p.saturatingAccumulation);
    }

    CoopMatrixConfig best;
    for (const auto &p : props)
    {
        const bool usable = p.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
                            p.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                            p.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
                            p.ResultType == VK_COMPONENT_TYPE_SINT32_KHR &&
                            p.scope == VK_SCOPE_SUBGROUP_KHR &&
                            p.saturatingAccumulation == VK_FALSE && p.MSize == p.NSize &&
                            kProblemM % p.MSize == 0 && kProblemN % p.NSize == 0 &&
                            kProblemK % p.KSize == 0;
        if (!usable)
            continue;

        const uint32_t area = p.MSize * p.NSize;
        if (!best.found || area > best.tileM * best.tileN ||
            (area == best.tileM * best.tileN && p.KSize > best.tileK))
        {
            best.found = true;
            best.tileM = p.MSize;
            best.tileN = p.NSize;
            best.tileK = p.KSize;
        }
    }
    best.subgroupSize = subgroupProps.subgroupSize;
    return best;
}

int main()
{
    try
    {
        // --- request the extension + its feature (best-effort) ---------------
        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures;
        coopFeatures.cooperativeMatrix = VK_TRUE;

        GpuContextDesc desc;
        // SPIR-V 1.6 (required by CooperativeMatrixKHR) needs Vulkan 1.3.
        desc.vulkanApiVersion = VK_API_VERSION_1_3;
        desc.deviceExtensions.push_back(
            {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, &continueWithoutCoopMatrix, &coopFeatures});

        GpuContext ctx(desc);
        std::printf("[cooperative_matrix] GPU: %s\n", ctx.deviceName().c_str());

        if (!ctx.hasDeviceExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
        {
            std::printf("[cooperative_matrix] VK_KHR_cooperative_matrix not supported; skipping\n");
            std::printf("%s\n", ctx.infoDump().c_str());
            return 0;
        }

        const CoopMatrixConfig cfg = findCoopMatrixConfig(ctx);
        if (!cfg.found)
        {
            std::printf("[cooperative_matrix] no usable cooperative-matrix configuration "
                        "(SINT8xSINT8->SINT32, subgroup scope, non-saturating, square tiles); "
                        "skipping\n");
            std::printf("%s\n", ctx.infoDump().c_str());
            return 0;
        }
        std::printf("[cooperative_matrix] using %ux%ux%u tiles, subgroup size %u\n", cfg.tileM,
                    cfg.tileN, cfg.tileK, cfg.subgroupSize);

        // --- data: C[32x32] = A[32x32] * B[32x32] --------------------------
        std::vector<int8_t> hostA(kProblemM * kProblemK), hostB(kProblemK * kProblemN);
        for (uint32_t i = 0; i < kProblemM; ++i)
            for (uint32_t k = 0; k < kProblemK; ++k)
                hostA[i * kProblemK + k] = static_cast<int8_t>((i * 3 + k * 5) % 7);
        for (uint32_t k = 0; k < kProblemK; ++k)
            for (uint32_t j = 0; j < kProblemN; ++j)
                hostB[k * kProblemN + j] = static_cast<int8_t>((k + j) % 3 + 1);

        std::vector<int32_t> refC(kProblemM * kProblemN, 0);
        for (uint32_t i = 0; i < kProblemM; ++i)
            for (uint32_t j = 0; j < kProblemN; ++j)
                for (uint32_t k = 0; k < kProblemK; ++k)
                    refC[i * kProblemN + j] += hostA[i * kProblemK + k] * hostB[k * kProblemN + j];

        GpuBuffer bufA(ctx, hostA.size(), GpuBuffer::Type::HostCoherent);
        GpuBuffer bufB(ctx, hostB.size(), GpuBuffer::Type::HostCoherent);
        GpuBuffer bufC(ctx, refC.size() * sizeof(int32_t), GpuBuffer::Type::HostCoherent);
        bufA.upload(std::span<const int8_t>(hostA));
        bufB.upload(std::span<const int8_t>(hostB));

        struct PC
        {
            uint32_t m, n, k, pad;
        } pc{kProblemM, kProblemN, kProblemK, 0};

        // --- compile the parameterized kernel for SPIR-V 1.6 -----------------
        ShaderOptions options;
        options.macros = {{"TILE_M", std::to_string(cfg.tileM)},
                          {"TILE_N", std::to_string(cfg.tileN)},
                          {"TILE_K", std::to_string(cfg.tileK)},
                          {"SUBGROUP_SIZE", std::to_string(cfg.subgroupSize)}};
        options.spirvProfile = "spirv_1_6";

        CompiledShader compiled = compileSlangToSpirV(kShader, "main", "", options);
        std::printf("[cooperative_matrix] SPIR-V %u words\n",
                    static_cast<uint32_t>(compiled.spirv.size()));

        GpuComputePipelineDesc pipeDesc;
        pipeDesc.spirv = compiled.spirv;
        pipeDesc.bindings = {bufA.descriptorInfo(), bufB.descriptorInfo(), bufC.descriptorInfo()};
        pipeDesc.pushConstantSize = sizeof(pc);
        pipeDesc.groupsX = cfg.subgroupSize;
        GpuComputePipeline pipeline(ctx, pipeDesc);

        GpuCommandBuffer cmd(ctx);
        cmd.begin();
        cmd.bind(pipeline);
        cmd.pushConstants(&pc, sizeof(pc));
        cmd.dispatch(kProblemN / cfg.tileN, kProblemM / cfg.tileM, 1);
        cmd.end();
        cmd.submitAndWait();

        std::vector<int32_t> hostC(kProblemM * kProblemN);
        bufC.download(std::span<int32_t>(hostC));

        bool ok = true;
        for (uint32_t i = 0; i < kProblemM && ok; ++i)
            for (uint32_t j = 0; j < kProblemN; ++j)
                if (hostC[i * kProblemN + j] != refC[i * kProblemN + j])
                {
                    std::printf("[cooperative_matrix] mismatch at [%u][%u]: got %d want %d\n", i, j,
                                hostC[i * kProblemN + j], refC[i * kProblemN + j]);
                    ok = false;
                    break;
                }

        std::printf("[cooperative_matrix] %ux%ux%u int8 matmul via cooperative matrix: %s\n",
                    kProblemM, kProblemN, kProblemK, ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "[cooperative_matrix] ERROR: %s\n", e.what());
        return 1;
    }
}
