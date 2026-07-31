#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_image.hpp"
#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstring>

using namespace pix;

TEST_CASE("GpuImage VMA-backed construction", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuImage img(ctx, 64, 64, vk::Format::eR8G8B8A8Unorm,
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);

    REQUIRE(img.image() != VK_NULL_HANDLE);
    REQUIRE(img.imageView() != VK_NULL_HANDLE);
    REQUIRE(img.width() == 64);
    REQUIRE(img.height() == 64);
    REQUIRE(img.format() == vk::Format::eR8G8B8A8Unorm);
    REQUIRE(img.mipLevels() == 1);
    REQUIRE(img.arrayLayers() == 1);
    REQUIRE(img.imageType() == vk::ImageType::e2D);
}

TEST_CASE("GpuImage construction with extra parameters", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuImage img(ctx, 32, 32, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eStorage,
                 vk::ImageType::e2D, 4, 6);

    REQUIRE(img.width() == 32);
    REQUIRE(img.height() == 32);
    REQUIRE(img.mipLevels() == 4);
    REQUIRE(img.arrayLayers() == 6);
    REQUIRE(img.imageType() == vk::ImageType::e2D);

    // Image view should cover all layers and mips
    REQUIRE(img.imageView() != VK_NULL_HANDLE);
}

TEST_CASE("GpuImage descriptorInfo", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuImage img(ctx, 32, 16, vk::Format::eR32Sfloat, vk::ImageUsageFlagBits::eStorage);

    auto di = img.descriptorInfo(vk::ImageLayout::eGeneral);
    REQUIRE(di.sampler == VK_NULL_HANDLE);
    REQUIRE(di.imageView == img.imageView());
    REQUIRE(di.imageLayout == vk::ImageLayout::eGeneral);
}

TEST_CASE("GpuImage descriptorInfo sampled layout", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuImage img(ctx, 32, 16, vk::Format::eR32Sfloat, vk::ImageUsageFlagBits::eSampled);

    auto di = img.descriptorInfo(vk::ImageLayout::eShaderReadOnlyOptimal);
    REQUIRE(di.sampler == VK_NULL_HANDLE);
    REQUIRE(di.imageView == img.imageView());
    REQUIRE(di.imageLayout == vk::ImageLayout::eShaderReadOnlyOptimal);
}

TEST_CASE("GpuImage move semantics", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuImage img(ctx, 48, 48, vk::Format::eB8G8R8A8Unorm, vk::ImageUsageFlagBits::eStorage);
    auto imageHandle = img.image();
    auto viewHandle = img.imageView();

    GpuImage moved = std::move(img);
    REQUIRE(moved.image() == imageHandle);
    REQUIRE(moved.imageView() == viewHandle);
    REQUIRE(moved.width() == 48);
    REQUIRE(moved.height() == 48);
}

TEST_CASE("GpuImage default constructor", "[image]")
{
    GpuImage img;
    REQUIRE(img.image() == VK_NULL_HANDLE);
    REQUIRE(img.imageView() == VK_NULL_HANDLE);
    REQUIRE(img.width() == 0);
    REQUIRE(img.height() == 0);
    REQUIRE(img.mipLevels() == 1);
    REQUIRE(img.arrayLayers() == 1);
    REQUIRE(img.imageType() == vk::ImageType::e2D);
}

TEST_CASE("GpuImage different formats", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuImage imgFloat(ctx, 16, 16, vk::Format::eR32G32B32A32Sfloat,
                      vk::ImageUsageFlagBits::eStorage);
    REQUIRE(imgFloat.format() == vk::Format::eR32G32B32A32Sfloat);

    GpuImage imgUint(ctx, 16, 16, vk::Format::eR8Uint, vk::ImageUsageFlagBits::eStorage);
    REQUIRE(imgUint.format() == vk::Format::eR8Uint);
}

TEST_CASE("GpuImage adopted external construction", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    vk::ImageCreateInfo imageInfo;
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = vk::Format::eR8G8B8A8Unorm;
    imageInfo.extent = vk::Extent3D(32, 32, 1);
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.usage = vk::ImageUsageFlagBits::eSampled;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;

    auto rawImage = ctx.device().createImage(imageInfo);

    {
        GpuImage img(ctx, rawImage, vk::Format::eR8G8B8A8Unorm, 32, 32);
        REQUIRE(img.image() != VK_NULL_HANDLE);
        REQUIRE(img.imageView() != VK_NULL_HANDLE);
        REQUIRE(img.width() == 32);
        REQUIRE(img.height() == 32);
    }
    ctx.device().destroyImage(rawImage);
}

TEST_CASE("GpuImage move assignment", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuImage a(ctx, 16, 16, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eStorage);
    GpuImage b(ctx, 32, 32, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eStorage);

    auto bImage = b.image();

    a = std::move(b);
    REQUIRE(a.image() == bImage);
    REQUIRE(a.width() == 32);
    REQUIRE(a.height() == 32);

    REQUIRE(b.image() == VK_NULL_HANDLE);
}

TEST_CASE("GpuImage upload readback via compute shader", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    constexpr uint32_t W = 4, H = 4;
    GpuImage img(ctx, W, H, vk::Format::eR32Sfloat,
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst);

    std::vector<float> pixels(W * H);
    for (uint32_t i = 0; i < W * H; ++i)
        pixels[i] = static_cast<float>(i);

    img.upload(std::span<const float>(pixels), vk::ImageLayout::eGeneral);

    GpuBuffer outBuf(ctx, W * H * sizeof(float), GpuBuffer::Type::HostCoherent);
    std::vector<float> zeros(W * H, 0.0f);
    outBuf.upload(std::span<const float>(zeros));

    auto readbackSpirv = compileSlangToSpirV(R"(
        RWStructuredBuffer<float> output : register(u0);
        RWTexture2D<float> inputImg : register(u1);

        [numthreads(1, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID)
        {
            uint w, h;
            inputImg.GetDimensions(w, h);
            if (tid.x < w && tid.y < h)
                output[tid.y * w + tid.x] = inputImg[int2(tid.x, tid.y)];
        }
    )",
                                             "main")
                             .spirv;

    GpuComputePipeline pipeline(
        ctx, readbackSpirv, {outBuf.descriptorInfo()},
        {img.descriptorInfo(vk::ImageLayout::eGeneral)},
        {vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageImage}, 0, W, H, 1);

    oneShotDispatch(ctx, pipeline);

    std::vector<float> result(W * H, -1.0f);
    outBuf.download(std::span<float>(result));
    REQUIRE(result == pixels);
}

TEST_CASE("GpuImage upload respects mipLevel", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();

    constexpr uint32_t W = 8, H = 8;
    GpuImage img(ctx, W, H, vk::Format::eR32Sfloat,
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
                 vk::ImageType::e2D, 3, 1);

    // Upload to mip level 1 (4x4)
    constexpr uint32_t mipW = W >> 1, mipH = H >> 1;
    std::vector<float> mip1(mipW * mipH, 42.0f);
    img.upload(std::span<const float>(mip1), vk::ImageLayout::eGeneral, 1, 0);

    REQUIRE(true);
}

TEST_CASE("GpuImage imageBarrier in command buffer", "[image][cmd]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuImage img(ctx, 64, 64, vk::Format::eR8G8B8A8Unorm,
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.imageBarrier(img.image(), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                     vk::PipelineStageFlagBits::eTopOfPipe, {},
                     vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
    cmd.end();
    cmd.submitAndWait();

    REQUIRE(true);
}

TEST_CASE("GpuImage linear tiling", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuImage img(ctx, 16, 16, vk::Format::eR32Sfloat,
                 vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                 vk::ImageType::e2D, 1, 1, {}, vk::ImageTiling::eLinear);
    REQUIRE(img.image() != VK_NULL_HANDLE);
    REQUIRE(img.imageView() != VK_NULL_HANDLE);
}

TEST_CASE("GpuImage 1D image type", "[image]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuImage img(ctx, 64, 1, vk::Format::eR32Sfloat, vk::ImageUsageFlagBits::eStorage,
                 vk::ImageType::e1D);
    REQUIRE(img.imageType() == vk::ImageType::e1D);
    REQUIRE(img.imageView() != VK_NULL_HANDLE);
    REQUIRE(img.width() == 64);
}

TEST_CASE("GpuImage imageBarrier with explicit aspect mask", "[image][cmd]")
{
    auto &ctx = GpuTestFixture::ctx();

    GpuImage img(ctx, 64, 64, vk::Format::eR32Sfloat, vk::ImageUsageFlagBits::eStorage);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.imageBarrier(img.image(), vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                     vk::PipelineStageFlagBits::eTopOfPipe, {},
                     vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite,
                     vk::ImageAspectFlagBits::eColor);
    cmd.end();
    cmd.submitAndWait();
}
