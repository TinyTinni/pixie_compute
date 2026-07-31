#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_image.hpp"
#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <unordered_map>

using namespace pix;

static const char *kSimpleShader = R"(
    [numthreads(4, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID)
    {
    }
)";

static const char *kBufferWriteShader = R"(
   RWStructuredBuffer<float> buffer : register(u0);

    [numthreads(4, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID)
    {
        buffer[tid.x] = float(tid.x) * 10.0;
    }
)";

static const char *kPushConstantShader = R"(
    [[vk::push_constant]]
    cbuffer PushConstants : register(b0)
    {
        float scale;
    };

    RWStructuredBuffer<float> buffer : register(u0);

    [numthreads(4, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID)
    {
        buffer[tid.x] = float(tid.x) * scale;
    }
)";

static const std::vector<uint32_t> &compile(const char *source)
{
    static std::unordered_map<std::string, std::vector<uint32_t>> s_cache;
    auto it = s_cache.find(source);
    if (it != s_cache.end())
        return it->second;
    auto result = compileSlangToSpirV(source, "main").spirv;
    auto [newIt, _] = s_cache.emplace(source, std::move(result));
    return newIt->second;
}

TEST_CASE("GpuComputePipeline construction", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);
    REQUIRE_FALSE(spirv.empty());

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();

    GpuComputePipeline pipeline(ctx, spirv, {di}, 0, 4, 1, 1);
    REQUIRE(pipeline.handle() != VK_NULL_HANDLE);
    REQUIRE(pipeline.layout() != VK_NULL_HANDLE);
    REQUIRE(pipeline.descriptorSetCount() == 1);
}

TEST_CASE("GpuComputePipeline dispatch writes to buffer", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kBufferWriteShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    std::vector<float> zeros(4, 0.0f);
    buf.upload(std::span<const float>(zeros));

    auto di = buf.descriptorInfo();
    GpuComputePipeline pipeline(ctx, spirv, {di}, 0, 4, 1, 1);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);
    cmd.dispatch(pipeline);
    cmd.end();
    cmd.submitAndWait();

    std::vector<float> result(4, 0.0f);
    buf.download(std::span<float>(result));
    REQUIRE(result[0] == 0.0f);
    REQUIRE(result[1] == 10.0f);
    REQUIRE(result[2] == 20.0f);
    REQUIRE(result[3] == 30.0f);
}

TEST_CASE("GpuComputePipeline push constants", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kPushConstantShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    std::vector<float> zeros(4, 0.0f);
    buf.upload(std::span<const float>(zeros));

    auto di = buf.descriptorInfo();
    float scale = 5.0f;
    GpuComputePipeline pipeline(ctx, spirv, {di}, sizeof(float), 4, 1, 1);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);
    cmd.pushConstants(pipeline, &scale, sizeof(float));
    cmd.dispatch(pipeline);
    cmd.end();
    cmd.submitAndWait();

    std::vector<float> result(4, 0.0f);
    buf.download(std::span<float>(result));
    REQUIRE(result[0] == 0.0f);
    REQUIRE(result[1] == 5.0f);
    REQUIRE(result[2] == 10.0f);
    REQUIRE(result[3] == 15.0f);
}

TEST_CASE("GpuComputePipeline auto-derives push constant size", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();

    GpuComputePipeline pipeline(ctx, kPushConstantShader, {di}, 0, 4, 1, 1);
    REQUIRE(pipeline.pushConstantSize() == sizeof(float));
}

TEST_CASE("GpuComputePipeline mismatched explicit push constant size throws", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();

    REQUIRE_THROWS_AS(
        GpuComputePipeline(ctx, kPushConstantShader, {di}, sizeof(float) * 2, 4, 1, 1), GpuError);
}

TEST_CASE("GpuComputePipeline updateBindings", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kBufferWriteShader);

    GpuBuffer buf1(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    GpuBuffer buf2(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    std::vector<float> zeros(4, 0.0f);
    buf1.upload(std::span<const float>(zeros));
    buf2.upload(std::span<const float>(zeros));

    auto di1 = buf1.descriptorInfo();
    GpuComputePipeline pipeline(ctx, spirv, {di1}, 0, 4, 1, 1);

    auto di2 = buf2.descriptorInfo();
    pipeline.updateBindings({di2});

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);
    cmd.dispatch(pipeline);
    cmd.end();
    cmd.submitAndWait();

    std::vector<float> result1(4, -1.0f);
    buf1.download(std::span<float>(result1));
    for (auto v : result1)
        REQUIRE(v == 0.0f);

    std::vector<float> result2(4, 0.0f);
    buf2.download(std::span<float>(result2));
    REQUIRE(result2[0] == 0.0f);
    REQUIRE(result2[1] == 10.0f);
    REQUIRE(result2[2] == 20.0f);
    REQUIRE(result2[3] == 30.0f);
}

TEST_CASE("GpuComputePipeline addDescriptorSet", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();
    GpuComputePipeline pipeline(ctx, spirv, {di}, 0, 4, 1, 1);
    REQUIRE(pipeline.descriptorSetCount() == 1);

    uint32_t idx = pipeline.addDescriptorSet({di});
    REQUIRE(idx == 1);
    REQUIRE(pipeline.descriptorSetCount() == 2);
}

TEST_CASE("GpuComputePipeline clearDescriptorSets", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();
    GpuComputePipeline pipeline(ctx, spirv, {di}, 0, 4, 1, 1);
    REQUIRE(pipeline.descriptorSetCount() == 1);

    pipeline.clearDescriptorSets();
    REQUIRE(pipeline.descriptorSetCount() == 0);
}

TEST_CASE("GpuComputePipeline move semantics", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();
    GpuComputePipeline pipeline(ctx, spirv, {di}, 0, 4, 1, 1);
    auto handle = pipeline.handle();

    GpuComputePipeline moved = std::move(pipeline);
    REQUIRE(moved.handle() == handle);
}

TEST_CASE("GpuComputePipeline PipelineDesc constructor", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();

    PipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings.push_back(di);
    desc.workgroupX = 4;
    desc.workgroupY = 1;
    desc.workgroupZ = 1;

    GpuComputePipeline pipeline(ctx, desc);
    REQUIRE(pipeline.handle() != VK_NULL_HANDLE);
    REQUIRE(pipeline.descriptorSetCount() == 1);
}

TEST_CASE("GpuComputePipeline from Slang source string", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();

    GpuComputePipeline pipeline(ctx, kBufferWriteShader, {di}, 0, 4, 1, 1);
    REQUIRE(pipeline.handle() != VK_NULL_HANDLE);

    oneShotDispatch(ctx, pipeline);

    std::vector<float> result(4, 0.0f);
    buf.download(std::span<float>(result));
    REQUIRE(result[0] == 0.0f);
    REQUIRE(result[1] == 10.0f);
    REQUIRE(result[2] == 20.0f);
    REQUIRE(result[3] == 30.0f);
}

TEST_CASE("GpuComputePipeline updateBindings validates buffer count", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();
    GpuComputePipeline pipeline(ctx, spirv, {di}, 0, 4, 1, 1);

    GpuBuffer buf2(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di2 = buf2.descriptorInfo();
    REQUIRE_THROWS_AS(pipeline.updateBindings({di, di2}), GpuError);
}

TEST_CASE("GpuComputePipeline mixed image+buffer construction", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();

    // Shader that reads from a storage image and writes to a buffer
    auto spirv = compile(R"(
        RWStructuredBuffer<float> output : register(u0);

        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID)
        {
            output[tid.x] = float(tid.x) + 1.0;
        }
    )");

    GpuBuffer outBuf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto bufDi = outBuf.descriptorInfo();

    // Create a dummy image for the eStorageImage binding
    GpuImage img(ctx, 4, 4, vk::Format::eR32Sfloat,
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);
    auto imgDi = img.descriptorInfo(vk::ImageLayout::eGeneral);

    std::vector<vk::DescriptorType> bindingTypes = {vk::DescriptorType::eStorageBuffer,
                                                    vk::DescriptorType::eStorageImage};

    GpuComputePipeline pipeline(ctx, spirv, {bufDi}, {imgDi}, bindingTypes, 0, 4, 1, 1);
    REQUIRE(pipeline.handle() != VK_NULL_HANDLE);
    REQUIRE(pipeline.descriptorSetCount() == 1);

    // Dispatch and verify
    oneShotDispatch(ctx, pipeline);

    std::vector<float> result(4, 0.0f);
    outBuf.download(std::span<float>(result));
    REQUIRE(result[0] == 1.0f);
    REQUIRE(result[1] == 2.0f);
    REQUIRE(result[2] == 3.0f);
    REQUIRE(result[3] == 4.0f);
}

TEST_CASE("GpuComputePipeline updateBindings mixed image+buffer", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();

    auto spirv = compile(R"(
        RWStructuredBuffer<float> output : register(u0);

        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID)
        {
            output[tid.x] = float(tid.x) * 5.0;
        }
    )");

    GpuBuffer outBuf1(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    GpuBuffer outBuf2(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    std::vector<float> zeros(4, 0.0f);
    outBuf1.upload(std::span<const float>(zeros));
    outBuf2.upload(std::span<const float>(zeros));

    GpuImage img(ctx, 4, 4, vk::Format::eR32Sfloat,
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);

    std::vector<vk::DescriptorType> bindingTypes = {vk::DescriptorType::eStorageBuffer,
                                                    vk::DescriptorType::eStorageImage};

    GpuComputePipeline pipeline(ctx, spirv, {outBuf1.descriptorInfo()},
                                {img.descriptorInfo(vk::ImageLayout::eGeneral)}, bindingTypes, 0, 4,
                                1, 1);

    // Update to point to outBuf2
    pipeline.updateBindings({outBuf2.descriptorInfo()},
                            {img.descriptorInfo(vk::ImageLayout::eGeneral)});

    oneShotDispatch(ctx, pipeline);

    // outBuf1 should still be zeros
    std::vector<float> r1(4, -1.0f);
    outBuf1.download(std::span<float>(r1));
    for (auto v : r1)
        REQUIRE(v == 0.0f);

    // outBuf2 should have the computed values
    std::vector<float> r2(4, 0.0f);
    outBuf2.download(std::span<float>(r2));
    REQUIRE(r2[0] == 0.0f);
    REQUIRE(r2[1] == 5.0f);
    REQUIRE(r2[2] == 10.0f);
    REQUIRE(r2[3] == 15.0f);
}

TEST_CASE("GpuComputePipeline buffer-only updateBindings uses correct descriptor type",
          "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();

    auto spirv = compile(kBufferWriteShader);

    GpuBuffer buf1(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    GpuBuffer buf2(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    std::vector<float> zeros(4, 0.0f);
    buf1.upload(std::span<const float>(zeros));

    auto di1 = buf1.descriptorInfo();
    GpuComputePipeline pipeline(ctx, spirv, {di1}, 0, 4, 1, 1);

    // updateBindings with buffer-only should still use eStorageBuffer
    auto di2 = buf2.descriptorInfo();
    pipeline.updateBindings({di2});

    oneShotDispatch(ctx, pipeline);

    // buf1 unchanged
    std::vector<float> r1(4, -1.0f);
    buf1.download(std::span<float>(r1));
    for (auto v : r1)
        REQUIRE(v == 0.0f);

    // buf2 has values
    std::vector<float> r2(4, 0.0f);
    buf2.download(std::span<float>(r2));
    REQUIRE(r2[0] == 0.0f);
    REQUIRE(r2[1] == 10.0f);
    REQUIRE(r2[2] == 20.0f);
    REQUIRE(r2[3] == 30.0f);
}
