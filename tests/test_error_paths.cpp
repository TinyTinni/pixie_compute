#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_image.hpp"
#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pix;

static const char *kPushConstantShader = R"(
    [[vk::push_constant]]
    cbuffer PC : register(b0) { float scale; };
    RWStructuredBuffer<float> buffer : register(u0);
    [numthreads(4, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID)
    {
        buffer[tid.x] = float(tid.x) * scale;
    }
)";

TEST_CASE("GpuBuffer upload exceeds buffer size", "[error]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 16, GpuBuffer::Type::HostCoherent);

    float vals[8] = {};
    REQUIRE_THROWS_AS(buf.upload(std::span<const float>(vals)), GpuError);
}

TEST_CASE("GpuBuffer download exceeds buffer size", "[error]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 16, GpuBuffer::Type::HostCoherent);

    float vals[8] = {};
    REQUIRE_THROWS_AS(buf.download(std::span<float>(vals)), GpuError);
}

TEST_CASE("GpuBuffer device download exceeds buffer size", "[error]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 64, GpuBuffer::Type::Device);

    float vals[32] = {};
    REQUIRE_THROWS_AS(buf.download(std::span<float>(vals)), GpuError);
}

TEST_CASE("GpuBufferSlice descriptorOffset exceeds slice size", "[error]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 64, GpuBuffer::Type::HostCoherent);
    GpuBufferSlice slice(buf, 0, 16);

    REQUIRE_THROWS_AS(slice.descriptorInfo(32, 4), GpuError);
}

TEST_CASE("GpuComputePipeline push constant exceeds layout range", "[error]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);
    auto di = buf.descriptorInfo();

    GpuComputePipeline pipeline(ctx, kPushConstantShader, {di}, sizeof(float), 4, 1, 1);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.bind(pipeline);

    double tooBig[2] = {};
    REQUIRE_THROWS_AS(pipeline.pushConstants(cmd.handle(), tooBig, sizeof(tooBig)), GpuError);
    cmd.end();
}

TEST_CASE("GpuCommandBuffer upload exceeds buffer size", "[error]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 16, GpuBuffer::Type::Device);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();

    float vals[8] = {};
    REQUIRE_THROWS_AS(cmd.upload(buf, std::span<const float>(vals)), GpuError);
    cmd.end();
}

TEST_CASE("compileSlangToSpirV fails on invalid source", "[error]")
{
    REQUIRE_THROWS_AS(compileSlangToSpirV("this is not valid slang code @#$%"), GpuError);
}

TEST_CASE("GpuBufferSlice upload exceeds parent buffer range", "[error]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 32, GpuBuffer::Type::HostCoherent);
    GpuBufferSlice slice(buf, 16, 8);

    float vals[8] = {};
    REQUIRE_THROWS_AS(slice.upload(std::span<const float>(vals)), GpuError);
}

TEST_CASE("GpuImage upload with invalid mipLevel throws", "[image][error]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuImage img(ctx, 8, 8, vk::Format::eR32Sfloat,
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
                 vk::ImageType::e2D, 2, 1);

    float data[16] = {};
    REQUIRE_THROWS_AS(img.upload(std::span<const float>(data), vk::ImageLayout::eGeneral, 5, 0),
                      GpuError);
}
