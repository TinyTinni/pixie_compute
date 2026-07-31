#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_stream.hpp"
#include "pixie_compute/utility.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace pix;

TEST_CASE("GpuStream commits ordered work", "[stream]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buffer(ctx, 4 * sizeof(uint32_t), GpuBuffer::Type::Device);
    std::vector<uint32_t> source = {1, 2, 3, 4};
    GpuStream stream(ctx);
    stream.upload(buffer, std::span<const uint32_t>(source));
    GpuEvent event = stream.commit();

    REQUIRE(event.valid());
    REQUIRE(event.value() == 1);

    std::vector<uint32_t> result(4, 0);
    stream.download(buffer, std::span<uint32_t>(result));
    stream.commit();
    stream.wait();
    REQUIRE(result == source);
}

TEST_CASE("GpuStream events synchronize separate streams", "[stream]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buffer(ctx, 4 * sizeof(uint32_t), GpuBuffer::Type::Device);
    std::vector<uint32_t> source = {7, 8, 9, 10};

    GpuStream producer(ctx);
    producer.upload(buffer, std::span<const uint32_t>(source));
    GpuEvent ready = producer.commit();

    GpuStream consumer(ctx);
    consumer.wait(ready);
    std::vector<uint32_t> result(4, 0);
    consumer.download(buffer, std::span<uint32_t>(result));
    consumer.commit();
    consumer.wait();
    producer.wait();

    REQUIRE(result == source);
}

TEST_CASE("GpuStream rejects empty events", "[stream]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuStream stream(ctx);
    REQUIRE_THROWS_AS(stream.wait(GpuEvent{}), GpuError);
}

TEST_CASE("GpuStream upload/download offsets are in bytes", "[stream]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buffer(ctx, 8 * sizeof(uint32_t), GpuBuffer::Type::Device);
    GpuStream stream(ctx);
    std::vector<uint32_t> source = {1, 2, 3, 4};
    stream.upload(buffer, std::span<const uint32_t>(source), 4 * sizeof(uint32_t));
    stream.commit();
    stream.wait();

    std::vector<uint32_t> result(8, 0);
    buffer.download(result);
    REQUIRE(result[0] == 0);
    REQUIRE(result[4] == 1);
    REQUIRE(result[7] == 4);
}

TEST_CASE("GpuStream synchronizes across a separate transfer queue", "[stream]")
{
    auto &ctx = GpuTestFixture::ctx();
    if (!ctx.hasSeparateTransferQueue())
        SKIP("no separate transfer queue");
    GpuBuffer buffer(ctx, 4 * sizeof(uint32_t), GpuBuffer::Type::Device);
    GpuStream transfer(ctx, GpuCommandBuffer::QueueType::Transfer);
    std::vector<uint32_t> source = {1, 2, 3, 4};
    transfer.upload(buffer, std::span<const uint32_t>(source));
    GpuEvent ready = transfer.commit();

    GpuStream compute(ctx);
    compute.wait(ready);
    std::vector<uint32_t> result(4, 0);
    compute.download(buffer, std::span<uint32_t>(result));
    compute.commit();
    compute.wait();
    transfer.wait();

    REQUIRE(result == source);
}
