#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_image.hpp"
#include "pixie_compute/gpu_kernel.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace pix;

#ifdef PIXIE_COMPUTE_HAS_SLANG
TEST_CASE("GpuKernel binds reflected resources by name", "[kernel]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buffer(ctx, 4 * sizeof(uint32_t), GpuBuffer::Type::Device);
    buffer.upload(std::vector<uint32_t>(4, 0));

    GpuComputePipelineDesc desc;
    desc.slangSource = R"(
        [[vk::binding(0, 0)]] RWStructuredBuffer<uint> data;
        [numthreads(4, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID)
        {
            data[id.x] = id.x + 1;
        }
    )";
    desc.groupsX = 1;
    GpuKernel kernel(ctx, desc);
    kernel.set("data", buffer);

    GpuCommandBuffer command(ctx);
    command.begin();
    kernel.launch(command);
    command.end();
    command.submitAndWait();

    std::vector<uint32_t> result(4);
    buffer.download(result);
    REQUIRE(result == std::vector<uint32_t>{1, 2, 3, 4});
}

TEST_CASE("GpuKernel reports missing named bindings", "[kernel]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuComputePipelineDesc desc;
    desc.slangSource = R"(
        [[vk::binding(0, 0)]] RWStructuredBuffer<uint> data;
        [numthreads(1, 1, 1)] void main() { data[0] = 1; }
    )";
    GpuKernel kernel(ctx, desc);
    REQUIRE_THROWS_AS(kernel.set("missing", GpuBinding(vk::DescriptorBufferInfo{})), GpuError);
}

TEST_CASE("GpuKernel binds images by name with a valid default layout", "[kernel]")
{
    auto &ctx = GpuTestFixture::ctx();
    constexpr uint32_t W = 4, H = 4;
    GpuImage img(
        ctx, GpuImageDesc{W, H, 1, vk::Format::eR32Sfloat,
                          vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst});
    std::vector<float> pixels(W * H);
    for (uint32_t i = 0; i < W * H; ++i)
        pixels[i] = static_cast<float>(i);
    img.upload(std::span<const float>(pixels), vk::ImageLayout::eGeneral);

    GpuBuffer outBuf(ctx, W * H * sizeof(float), GpuBuffer::Type::HostCoherent);
    outBuf.upload(std::vector<float>(W * H, 0.0f));

    GpuComputePipelineDesc desc;
    desc.slangSource = R"(
        [[vk::binding(0, 0)]] RWStructuredBuffer<float> output;
        [[vk::binding(1, 0)]] RWTexture2D<float> inputImg;
        [numthreads(1, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID)
        {
            uint w, h;
            inputImg.GetDimensions(w, h);
            if (tid.x < w && tid.y < h)
                output[tid.y * w + tid.x] = inputImg[int2(tid.x, tid.y)];
        }
    )";
    desc.groupsX = W;
    desc.groupsY = H;
    GpuKernel kernel(ctx, desc);
    kernel.set("inputImg", img);
    kernel.set("output", outBuf);

    GpuCommandBuffer command(ctx);
    command.begin();
    kernel.launch(command, W, H);
    command.end();
    command.submitAndWait();

    std::vector<float> result(W * H, -1.0f);
    outBuf.download(result);
    REQUIRE(result == pixels);
}

TEST_CASE("GpuKernel rejects updates after descriptor sets cleared", "[kernel]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buffer(ctx, 4 * sizeof(uint32_t), GpuBuffer::Type::Device);
    GpuComputePipelineDesc desc;
    desc.slangSource = R"(
        [[vk::binding(0, 0)]] RWStructuredBuffer<uint> data;
        [numthreads(1, 1, 1)] void main() { data[0] = 1; }
    )";
    GpuKernel kernel(ctx, desc);
    kernel.pipeline().clearBindingSets();
    REQUIRE_FALSE(kernel.pipeline().bindingsComplete());
    REQUIRE_THROWS_AS(kernel.set("data", buffer), GpuError);
}

TEST_CASE("GpuKernel rejects buffer bound to an image binding", "[kernel]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buffer(ctx, 4, GpuBuffer::Type::Device);
    GpuComputePipelineDesc desc;
    desc.slangSource = R"(
        [[vk::binding(0, 0)]] RWTexture2D<float> img;
        [numthreads(1, 1, 1)] void main() { img[uint2(0, 0)] = 1.0; }
    )";
    GpuKernel kernel(ctx, desc);
    REQUIRE_THROWS_AS(kernel.set("img", buffer), GpuError);
}

TEST_CASE("GpuKernel rejects image bound to a buffer binding", "[kernel]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuImage img(ctx,
                 GpuImageDesc{1, 1, 1, vk::Format::eR32Sfloat, vk::ImageUsageFlagBits::eStorage});
    GpuComputePipelineDesc desc;
    desc.slangSource = R"(
        [[vk::binding(0, 0)]] RWStructuredBuffer<float> data;
        [numthreads(1, 1, 1)] void main() { data[0] = 1.0; }
    )";
    GpuKernel kernel(ctx, desc);
    REQUIRE_THROWS_AS(kernel.set("data", img), GpuError);
}

TEST_CASE("GpuKernel setPushConstants requires exact shader block size", "[kernel]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuComputePipelineDesc desc;
    desc.slangSource = R"(
        [[vk::binding(0, 0)]] RWStructuredBuffer<float> data;
        struct Push { float scale; uint offset; };
        [[vk::push_constant]] Push pc;
        [numthreads(1, 1, 1)] void main() { data[pc.offset] = pc.scale; }
    )";
    GpuKernel kernel(ctx, desc);
    REQUIRE(kernel.pipeline().pushConstantSize() == 8);

    struct Push
    {
        float scale;
        uint32_t offset;
    };
    Push pc{2.0f, 0u};
    kernel.setPushConstants(pc);
    REQUIRE_THROWS_AS(kernel.setPushConstants(&pc, sizeof(float)), GpuError);
    REQUIRE_THROWS_AS(kernel.setPushConstants(&pc, sizeof(Push) + 4), GpuError);
}

TEST_CASE("GpuKernel setPushConstants rejects kernels without a push-constant block", "[kernel]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuComputePipelineDesc desc;
    desc.slangSource = R"(
        [[vk::binding(0, 0)]] RWStructuredBuffer<float> data;
        [numthreads(1, 1, 1)] void main() { data[0] = 1.0; }
    )";
    GpuKernel kernel(ctx, desc);
    float value = 1.0f;
    REQUIRE_THROWS_AS(kernel.setPushConstants(&value, sizeof(float)), GpuError);
}
#endif
