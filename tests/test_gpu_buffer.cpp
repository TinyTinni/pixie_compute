#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>

using namespace pix;

TEST_CASE("GpuBuffer host-coherent upload/download round-trip", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);

    buf.upload(std::span<const float>(data));

    std::vector<float> result(4, 0.0f);
    buf.download(std::span<float>(result));
    REQUIRE(result == data);
}

TEST_CASE("GpuBuffer span-based upload/download", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data = {10.0f, 20.0f, 30.0f};
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);

    buf.upload(std::span<const float>(data));

    std::vector<float> result(3, 0.0f);
    buf.download(std::span<float>(result));
    REQUIRE(result == data);
}

TEST_CASE("GpuBuffer upload with offset", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> initial = {0.0f, 0.0f, 0.0f, 0.0f};
    GpuBuffer buf(ctx, initial.size() * sizeof(float), GpuBuffer::Type::HostCoherent);
    buf.upload(std::span<const float>(initial));

    std::vector<float> patch = {99.0f, 100.0f};
    buf.upload(std::span<const float>(patch), 2 * sizeof(float));

    std::vector<float> result(4, 0.0f);
    buf.download(std::span<float>(result));
    REQUIRE(result[0] == 0.0f);
    REQUIRE(result[1] == 0.0f);
    REQUIRE(result[2] == 99.0f);
    REQUIRE(result[3] == 100.0f);
}

TEST_CASE("GpuBuffer device upload/download round-trip via unified API", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> data = {10, 20, 30, 40};
    GpuBuffer buf(ctx, data.size() * sizeof(uint32_t), GpuBuffer::Type::Device);

    buf.upload(std::span<const uint32_t>(data));

    std::vector<uint32_t> result(4, 0);
    buf.download(std::span<uint32_t>(result));
    REQUIRE(result == data);
}

TEST_CASE("GpuBuffer device upload/download with offset", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> zeros(6, 0);
    GpuBuffer buf(ctx, zeros.size() * sizeof(uint32_t), GpuBuffer::Type::Device);
    buf.upload(std::span<const uint32_t>(zeros));

    std::vector<uint32_t> patch = {99, 100};
    buf.upload(std::span<const uint32_t>(patch), 2 * sizeof(uint32_t));

    std::vector<uint32_t> result(6, 0);
    buf.download(std::span<uint32_t>(result));
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 0);
    REQUIRE(result[2] == 99);
    REQUIRE(result[3] == 100);
    REQUIRE(result[4] == 0);
    REQUIRE(result[5] == 0);
}

TEST_CASE("GpuBufferSlice descriptorInfo", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);
    buf.upload(std::span<const float>(data));

    GpuBufferSlice slice(buf, sizeof(float) * 2, sizeof(float) * 2);
    REQUIRE(slice.offset() == sizeof(float) * 2);
    REQUIRE(slice.size() == sizeof(float) * 2);
    REQUIRE(slice.handle() == buf.handle());

    auto di = slice.descriptorInfo();
    REQUIRE(di.offset == sizeof(float) * 2);
    REQUIRE(di.range == sizeof(float) * 2);
}

TEST_CASE("GpuBufferSlice descriptorInfo with custom range", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data(8, 1.0f);
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);

    GpuBufferSlice slice(buf, sizeof(float), sizeof(float) * 6);
    auto di = slice.descriptorInfo(0, sizeof(float) * 2);
    REQUIRE(di.offset == sizeof(float));
    REQUIRE(di.range == sizeof(float) * 2);
}

TEST_CASE("GpuCommandBuffer download via GpuBufferSlice offset", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);
    buf.upload(std::span<const float>(data));

    GpuBufferSlice slice(buf, sizeof(float), sizeof(float) * 2);
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    std::vector<float> result(2, 0.0f);
    cmd.download(buf, std::span<float>(result), slice.offset());
    cmd.submitAndWait();
    REQUIRE(result[0] == 2.0f);
    REQUIRE(result[1] == 3.0f);
}

TEST_CASE("GpuCommandBuffer upload via GpuBufferSlice offset", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> initial = {0.0f, 0.0f, 0.0f, 0.0f};
    GpuBuffer buf(ctx, initial.size() * sizeof(float), GpuBuffer::Type::HostCoherent);
    buf.upload(std::span<const float>(initial));

    GpuBufferSlice slice(buf, sizeof(float), sizeof(float) * 2);
    std::vector<float> patch = {7.0f, 8.0f};
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.upload(buf, std::span<const float>(patch), slice.offset());
    cmd.submitAndWait();

    std::vector<float> result(4, 0.0f);
    buf.download(std::span<float>(result));
    REQUIRE(result[0] == 0.0f);
    REQUIRE(result[1] == 7.0f);
    REQUIRE(result[2] == 8.0f);
    REQUIRE(result[3] == 0.0f);
}

TEST_CASE("GpuBuffer move semantics", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data = {1.0f, 2.0f};
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);
    buf.upload(std::span<const float>(data));

    GpuBuffer moved = std::move(buf);
    std::vector<float> result(2, 0.0f);
    moved.download(std::span<float>(result));
    REQUIRE(result == data);
}

TEST_CASE("GpuBuffer device move semantics", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> data = {42, 43};
    GpuBuffer buf(ctx, data.size() * sizeof(uint32_t), GpuBuffer::Type::Device);
    buf.upload(std::span<const uint32_t>(data));

    GpuBuffer moved = std::move(buf);
    std::vector<uint32_t> result(2, 0);
    moved.download(std::span<uint32_t>(result));
    REQUIRE(result == data);
}

TEST_CASE("RapidCheck: buffer round-trip preserves arbitrary float data", "[rapidcheck][buffer]")
{
    RC_ASSERT(rc::check(
        [](std::vector<float> data)
        {
            RC_PRE(!data.empty());
            RC_PRE(data.size() <= 4096);

            auto &ctx = GpuTestFixture::ctx();
            GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);
            buf.upload(std::span<const float>(data));

            std::vector<float> result(data.size(), 0.0f);
            buf.download(std::span<float>(result));
            RC_ASSERT(result == data);
        }));
}

TEST_CASE("RapidCheck: buffer offset write preserves surrounding data", "[rapidcheck][buffer]")
{
    RC_ASSERT(rc::check(
        [](uint32_t val1, uint32_t val2)
        {
            auto &ctx = GpuTestFixture::ctx();
            std::vector<uint32_t> initial = {0, 0, 0, 0};
            GpuBuffer buf(ctx, initial.size() * sizeof(uint32_t), GpuBuffer::Type::HostCoherent);
            buf.upload(std::span<const uint32_t>(initial));

            buf.upload(std::span<const uint32_t>(&val1, 1), sizeof(uint32_t));
            buf.upload(std::span<const uint32_t>(&val2, 1), 2 * sizeof(uint32_t));

            std::vector<uint32_t> result(4, 99);
            buf.download(std::span<uint32_t>(result));
            RC_ASSERT(result[0] == 0);
            RC_ASSERT(result[1] == val1);
            RC_ASSERT(result[2] == val2);
            RC_ASSERT(result[3] == 0);
        }));
}

TEST_CASE("GpuBufferSlice upload and download", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);
    buf.upload(std::span<const float>(data));

    GpuBufferSlice slice(buf, sizeof(float), sizeof(float) * 2);

    std::vector<float> result(2, 0.0f);
    slice.download(std::span<float>(result));
    REQUIRE(result[0] == 2.0f);
    REQUIRE(result[1] == 3.0f);

    std::vector<float> patch = {5.0f, 6.0f};
    slice.upload(std::span<const float>(patch));

    std::vector<float> full(4, 0.0f);
    buf.download(std::span<float>(full));
    REQUIRE(full[0] == 1.0f);
    REQUIRE(full[1] == 5.0f);
    REQUIRE(full[2] == 6.0f);
    REQUIRE(full[3] == 4.0f);
}

TEST_CASE("GpuBufferSlice mapped<T> aligned access", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, sizeof(uint32_t) * 6, GpuBuffer::Type::HostCoherent);
    {
        auto span = buf.mapped<uint32_t>();
        for (size_t i = 0; i < span.size(); ++i)
            span[i] = static_cast<uint32_t>(i * 10);
    }

    GpuBufferSlice slice(buf, sizeof(uint32_t) * 2, sizeof(uint32_t) * 2);
    auto span = slice.mapped<uint32_t>();
    REQUIRE(span.size() == 2);
    REQUIRE(span[0] == 20);
    REQUIRE(span[1] == 30);
    span[0] = 99;
    REQUIRE(buf.mapped<uint32_t>()[2] == 99);
}

TEST_CASE("GpuBufferSlice mapped<T> unaligned offset throws", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 16, GpuBuffer::Type::HostCoherent);
    GpuBufferSlice slice(buf, 2, 8);
    REQUIRE_THROWS_AS(slice.mapped<uint32_t>(), GpuError);
}

TEST_CASE("GpuBufferSlice mapped<T> unaligned size throws", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 16, GpuBuffer::Type::HostCoherent);
    GpuBufferSlice slice(buf, 0, 6);
    REQUIRE_THROWS_AS(slice.mapped<uint32_t>(), GpuError);
}

TEST_CASE("GpuBufferSlice mapped<T> range beyond parent throws", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 12, GpuBuffer::Type::HostCoherent);
    GpuBufferSlice slice(buf, 8, 16);
    REQUIRE_THROWS_AS(slice.mapped<uint32_t>(), GpuError);
}

TEST_CASE("GpuBuffer HostCoherent memory is host-coherent", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 64, GpuBuffer::Type::HostCoherent);
    REQUIRE((buf.memoryProperties() & vk::MemoryPropertyFlagBits::eHostVisible) !=
            vk::MemoryPropertyFlags{});
    REQUIRE((buf.memoryProperties() & vk::MemoryPropertyFlagBits::eHostCoherent) !=
            vk::MemoryPropertyFlags{});
}

TEST_CASE("GpuBuffer clear on tiny device buffer throws", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 2, GpuBuffer::Type::Device);
    REQUIRE_THROWS_AS(buf.clear(), GpuError);
}

// ---------------------------------------------------------------------------
// Command buffer transfer tests (replaces old GpuTransfer tests)
// ---------------------------------------------------------------------------

TEST_CASE("GpuCommandBuffer host-coherent upload via command buffer", "[buffer][cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);

    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.upload(buf, std::span<const float>(data));
    cmd.submitAndWait();

    std::vector<float> result(3, 0.0f);
    buf.download(std::span<float>(result));
    REQUIRE(result == data);
}

TEST_CASE("GpuCommandBuffer host-coherent download via command buffer", "[buffer][cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data = {10.0f, 20.0f, 30.0f};
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::HostCoherent);
    buf.upload(std::span<const float>(data));

    std::vector<float> result(3, 0.0f);
    GpuCommandBuffer cmd(ctx);
    cmd.begin();
    cmd.download(buf, std::span<float>(result));
    cmd.submitAndWait();
    REQUIRE(result == data);
}

TEST_CASE("GpuCommandBuffer device upload/download round-trip", "[buffer][cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> data = {100, 200, 300, 400};
    GpuBuffer buf(ctx, data.size() * sizeof(uint32_t), GpuBuffer::Type::Device);

    GpuCommandBuffer upCmd(ctx, GpuCommandBuffer::QueueType::Transfer);
    upCmd.begin();
    upCmd.upload(buf, std::span<const uint32_t>(data));
    upCmd.submitAndWait();

    std::vector<uint32_t> result(4, 0);
    GpuCommandBuffer downCmd(ctx, GpuCommandBuffer::QueueType::Transfer);
    downCmd.begin();
    downCmd.download(buf, std::span<uint32_t>(result));
    downCmd.submitAndWait();
    REQUIRE(result == data);
}

TEST_CASE("GpuCommandBuffer device upload non-blocking", "[buffer][cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> data = {55, 66, 77};
    GpuBuffer buf(ctx, data.size() * sizeof(uint32_t), GpuBuffer::Type::Device);

    GpuCommandBuffer cmd(ctx, GpuCommandBuffer::QueueType::Transfer);
    cmd.begin();
    cmd.upload(buf, std::span<const uint32_t>(data));
    cmd.submit();
    cmd.wait();

    std::vector<uint32_t> result(3, 0);
    buf.download(std::span<uint32_t>(result));
    REQUIRE(result == data);
}

TEST_CASE("GpuCommandBuffer device download non-blocking", "[buffer][cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> data = {11, 22, 33};
    GpuBuffer buf(ctx, data.size() * sizeof(uint32_t), GpuBuffer::Type::Device);
    buf.upload(std::span<const uint32_t>(data));

    std::vector<uint32_t> result(3, 0);
    GpuCommandBuffer cmd(ctx, GpuCommandBuffer::QueueType::Transfer);
    cmd.begin();
    cmd.download(buf, std::span<uint32_t>(result));
    cmd.submit();
    cmd.wait();
    REQUIRE(result == data);
}

TEST_CASE("GpuCommandBuffer span-based upload/download", "[buffer][cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<float> data = {1.5f, 2.5f, 3.5f};
    GpuBuffer buf(ctx, data.size() * sizeof(float), GpuBuffer::Type::Device);

    GpuCommandBuffer upCmd(ctx, GpuCommandBuffer::QueueType::Transfer);
    upCmd.begin();
    upCmd.upload(buf, std::span<const float>(data));
    upCmd.submitAndWait();

    std::vector<float> result(3, 0.0f);
    GpuCommandBuffer downCmd(ctx, GpuCommandBuffer::QueueType::Transfer);
    downCmd.begin();
    downCmd.download(buf, std::span<float>(result));
    downCmd.submitAndWait();
    REQUIRE(result == data);
}

TEST_CASE("GpuCommandBuffer batch upload in one command buffer", "[buffer][cmd]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> dataA = {1, 2, 3};
    std::vector<uint32_t> dataB = {4, 5, 6};
    GpuBuffer bufA(ctx, dataA.size() * sizeof(uint32_t), GpuBuffer::Type::Device);
    GpuBuffer bufB(ctx, dataB.size() * sizeof(uint32_t), GpuBuffer::Type::Device);

    GpuCommandBuffer cmd(ctx, GpuCommandBuffer::QueueType::Transfer);
    cmd.begin();
    cmd.upload(bufA, std::span<const uint32_t>(dataA));
    cmd.upload(bufB, std::span<const uint32_t>(dataB));
    cmd.submitAndWait();

    std::vector<uint32_t> resultA(3, 0), resultB(3, 0);
    bufA.download(std::span<uint32_t>(resultA));
    bufB.download(std::span<uint32_t>(resultB));
    REQUIRE(resultA == dataA);
    REQUIRE(resultB == dataB);
}

// ---------------------------------------------------------------------------
// Transfer queue tests
// ---------------------------------------------------------------------------

TEST_CASE("GpuContext has separate transfer queue info", "[context][transfer]")
{
    auto &ctx = GpuTestFixture::ctx();
    bool hasTransfer = ctx.hasSeparateTransferQueue();
    (void)hasTransfer;
    REQUIRE_FALSE(ctx.transferQueue() == vk::Queue{});
    REQUIRE_FALSE(ctx.transferCommandPool() == vk::CommandPool{});
}

TEST_CASE("GpuCommandBuffer transfer queue type", "[cmd][transfer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuCommandBuffer cmd(ctx, GpuCommandBuffer::QueueType::Transfer);
    cmd.begin();
    REQUIRE(cmd.recording());
    cmd.end();
    cmd.submitAndWait();
}

TEST_CASE("GpuBuffer mapped<T> access", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, sizeof(float) * 4, GpuBuffer::Type::HostCoherent);

    auto span = buf.mapped<float>();
    REQUIRE(span.size() == 4);
    span[0] = 10.0f;
    span[1] = 20.0f;
    span[2] = 30.0f;
    span[3] = 40.0f;

    std::vector<float> result(4, 0.0f);
    buf.download(std::span<float>(result));
    REQUIRE(result[0] == 10.0f);
    REQUIRE(result[3] == 40.0f);

    const auto &cbuf = buf;
    auto cspan = cbuf.mapped<float>();
    REQUIRE(cspan[1] == 20.0f);
}

TEST_CASE("GpuBuffer with extra usage flags", "[buffer]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buf(ctx, 64, GpuBuffer::Type::HostCoherent, vk::BufferUsageFlagBits::eVertexBuffer);
    REQUIRE(buf.handle() != VK_NULL_HANDLE);
    REQUIRE(buf.size() == 64);
}
