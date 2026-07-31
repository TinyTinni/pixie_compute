#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_timeline_semaphore.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>

using namespace pix;

TEST_CASE("GpuDownload asynchronously reads a device buffer", "[download]")
{
    auto &ctx = GpuTestFixture::ctx();
    std::vector<uint32_t> source = {5, 10, 15, 20};
    GpuBuffer buffer(ctx, source.size() * sizeof(uint32_t), GpuBuffer::Type::Device);
    buffer.upload(source);

    auto download = buffer.downloadAsync(source.size() * sizeof(uint32_t));
    REQUIRE(download.valid());
    REQUIRE(download.getAs<uint32_t>() == source);
}

TEST_CASE("GpuDownload handles host-coherent buffers immediately", "[download]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buffer(ctx, 4 * sizeof(float), GpuBuffer::Type::HostCoherent);
    std::vector<float> source = {1.0f, 2.0f, 3.0f, 4.0f};
    buffer.upload(source);

    auto download = buffer.downloadAsync(2 * sizeof(float), sizeof(float));
    auto bytes = download.get();
    std::vector<float> result(2);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    REQUIRE(result == std::vector<float>{2.0f, 3.0f});
}

TEST_CASE("GpuDownload waits on an explicit timeline signal", "[download]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuBuffer buffer(ctx, 4 * sizeof(uint32_t), GpuBuffer::Type::Device);
    GpuTimelineSemaphore sem(ctx);
    const uint64_t val = sem.next();
    {
        GpuCommandBuffer cmd(ctx);
        cmd.begin();
        cmd.fillBuffer(buffer.handle(), 0xDEADBEEF, buffer.size());
        cmd.end();
        cmd.submit({.signals = {GpuSignalSemaphore::makeTimeline(sem, val)}});
        cmd.wait();
    }
    auto download = buffer.downloadAsync(4 * sizeof(uint32_t), 0,
                                         {.waits = {GpuWaitSemaphore::makeTimeline(sem, val)}});
    REQUIRE(download.getAs<uint32_t>() == std::vector<uint32_t>(4, 0xDEADBEEF));
}
