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
   [[vk::binding(0, 0)]] RWStructuredBuffer<float> buffer;

    [numthreads(4, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID)
    {
        buffer[tid.x] = float(tid.x) * 10.0;
    }
)";

static const char *kPushConstantShader = R"(
    [[vk::push_constant]]
    cbuffer PushConstants
    {
        float scale;
    };

    [[vk::binding(0, 0)]] RWStructuredBuffer<float> buffer;

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

    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {di};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);
    REQUIRE(pipeline.handle() != VK_NULL_HANDLE);
    REQUIRE(pipeline.layout() != VK_NULL_HANDLE);
    REQUIRE(pipeline.bindingSetCount() == 1);
}

TEST_CASE("GpuComputePipeline dispatch writes to buffer", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kBufferWriteShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    std::vector<float> zeros(4, 0.0f);
    buf.upload(std::span<const float>(zeros));

    auto di = buf.descriptorInfo();
    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {di};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);
    cmd.dispatch();
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
    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {di};
    desc.pushConstantSize = sizeof(float);
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);
    cmd.pushConstants(&scale, sizeof(float));
    cmd.dispatch();
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

    GpuComputePipelineDesc desc;
    desc.slangSource = kPushConstantShader;
    desc.bindings = {di};
    GpuComputePipeline pipeline(ctx, desc);
    REQUIRE(pipeline.pushConstantSize() == sizeof(float));
}

TEST_CASE("GpuComputePipeline mismatched explicit push constant size throws", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();

    GpuComputePipelineDesc desc;
    desc.slangSource = kPushConstantShader;
    desc.bindings = {di};
    desc.pushConstantSize = sizeof(float) * 2;
    desc.groupsX = 4;
    REQUIRE_THROWS_AS(GpuComputePipeline(ctx, desc), GpuError);
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
    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {di1};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);

    auto di2 = buf2.descriptorInfo();
    pipeline.updateBindings({di2});

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);
    cmd.dispatch();
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

TEST_CASE("GpuComputePipeline addBindingSet", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();
    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {di};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);
    REQUIRE(pipeline.bindingSetCount() == 1);

    uint32_t idx = pipeline.addBindingSet({di});
    REQUIRE(idx == 1);
    REQUIRE(pipeline.bindingSetCount() == 2);
}

TEST_CASE("GpuComputePipeline clearBindingSets", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();
    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {di};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);
    REQUIRE(pipeline.bindingSetCount() == 1);

    pipeline.clearBindingSets();
    REQUIRE(pipeline.bindingSetCount() == 0);
}

TEST_CASE("GpuComputePipeline move semantics", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();
    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {di};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);
    auto handle = pipeline.handle();

    GpuComputePipeline moved = std::move(pipeline);
    REQUIRE(moved.handle() == handle);
}

TEST_CASE("GpuComputePipeline desc constructor", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();
    auto spirv = compile(kSimpleShader);

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();

    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings.push_back(di);
    desc.groupsX = 4;

    GpuComputePipeline pipeline(ctx, desc);
    REQUIRE(pipeline.handle() != VK_NULL_HANDLE);
    REQUIRE(pipeline.bindingSetCount() == 1);
}

TEST_CASE("GpuComputePipeline from Slang source string", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();

    GpuComputePipelineDesc desc;
    desc.slangSource = kBufferWriteShader;
    desc.bindings = {di};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);
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
    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {di};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);

    GpuBuffer buf2(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di2 = buf2.descriptorInfo();
    REQUIRE_THROWS_AS(pipeline.updateBindings({di, di2}), GpuError);
}

TEST_CASE("GpuComputePipeline mixed image+buffer construction", "[pipeline]")
{
    auto &ctx = GpuTestFixture::ctx();

    // Shader that reads from a storage image and writes to a buffer
    auto spirv = compile(R"(
        [[vk::binding(0, 0)]] RWStructuredBuffer<float> output;

        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID)
        {
            output[tid.x] = float(tid.x) + 1.0;
        }
    )");

    GpuBuffer outBuf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto bufDi = outBuf.descriptorInfo();

    // Create a dummy image for the eStorageImage binding
    GpuImage img(ctx, GpuImageDesc{4, 4, 1, vk::Format::eR32Sfloat,
                                   vk::ImageUsageFlagBits::eStorage |
                                       vk::ImageUsageFlagBits::eSampled});
    auto imgDi = img.descriptorInfo(vk::ImageLayout::eGeneral);

    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {bufDi, GpuBinding(imgDi, vk::DescriptorType::eStorageImage)};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);
    REQUIRE(pipeline.handle() != VK_NULL_HANDLE);
    REQUIRE(pipeline.bindingSetCount() == 1);

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
        [[vk::binding(0, 0)]] RWStructuredBuffer<float> output;

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

    GpuImage img(ctx, GpuImageDesc{4, 4, 1, vk::Format::eR32Sfloat,
                                   vk::ImageUsageFlagBits::eStorage |
                                       vk::ImageUsageFlagBits::eSampled});

    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {outBuf1.descriptorInfo(),
                     GpuBinding(img.descriptorInfo(vk::ImageLayout::eGeneral),
                                vk::DescriptorType::eStorageImage)};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);

    // Update to point to outBuf2
    pipeline.updateBindings({outBuf2.descriptorInfo(),
                             GpuBinding(img.descriptorInfo(vk::ImageLayout::eGeneral),
                                        vk::DescriptorType::eStorageImage)});

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
    GpuComputePipelineDesc desc;
    desc.spirv = spirv;
    desc.bindings = {di1};
    desc.groupsX = 4;
    GpuComputePipeline pipeline(ctx, desc);

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
