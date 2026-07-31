#include "pixie_compute/gpu_vector.hpp"
#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace pix;

TEST_CASE("GpuVector host-coherent typed round trip", "[vector]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuVector<uint32_t> values(ctx, 4, GpuBuffer::Type::HostCoherent);
    values.assign(std::vector<uint32_t>{1, 2, 3, 4});

    std::vector<uint32_t> result(4);
    values.download(result);
    REQUIRE(result == std::vector<uint32_t>{1, 2, 3, 4});
    REQUIRE(values.descriptorInfo().range == 4 * sizeof(uint32_t));
}

TEST_CASE("GpuVector device upload and resize preserve prefix", "[vector]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuVector<uint32_t> values(ctx, 3, GpuBuffer::Type::Device);
    values.assign(std::vector<uint32_t>{10, 20, 30});

    values.resize(5);
    std::vector<uint32_t> result(5, 0);
    values.download(result);
    REQUIRE(result[0] == 10);
    REQUIRE(result[1] == 20);
    REQUIRE(result[2] == 30);

    values.upload(std::vector<uint32_t>{99, 100}, 3);
    values.download(result);
    REQUIRE(result == std::vector<uint32_t>{10, 20, 30, 99, 100});
}

TEST_CASE("GpuVector empty state and range validation", "[vector]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuVector<float> values(ctx);
    REQUIRE(values.empty());
    REQUIRE(values.mapped().empty());
    REQUIRE_THROWS_AS(values.descriptorInfo(), GpuError);
    REQUIRE_THROWS_AS(values.upload(std::span<const float>{}), GpuError);

    values.resize(2);
    REQUIRE(values.size() == 2);
    values.resize(0);
    REQUIRE(values.empty());
}
