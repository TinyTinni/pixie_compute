#include "pixie_compute/gpu_tensor.hpp"

#include "test_fixture.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

using namespace pix;

TEST_CASE("GpuTensor tracks runtime shape and row-major strides", "[tensor]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuTensor<float, 2> tensor(ctx, {3, 4}, GpuBuffer::Type::HostCoherent);
    REQUIRE(tensor.elements() == 12);
    REQUIRE(tensor.shape() == std::array<size_t, 2>{3, 4});
    REQUIRE(tensor.strides() == std::array<size_t, 2>{4, 1});

    auto reshaped = tensor.reshape({2, 6});
    REQUIRE(reshaped.shape() == std::array<size_t, 2>{2, 6});
    REQUIRE(reshaped.strides() == std::array<size_t, 2>{6, 1});
}

TEST_CASE("GpuTensor typed upload and download", "[tensor]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuTensor<uint32_t, 3> tensor(ctx, {2, 2, 2}, GpuBuffer::Type::Device);
    std::vector<uint32_t> values = {1, 2, 3, 4, 5, 6, 7, 8};
    tensor.upload(values);

    std::vector<uint32_t> result(values.size());
    tensor.download(result);
    REQUIRE(result == values);
}

TEST_CASE("GpuTensor rejects mismatched transfers and reshapes", "[tensor]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuTensor<float, 2> tensor(ctx, {2, 3}, GpuBuffer::Type::HostCoherent);
    REQUIRE_THROWS_AS(tensor.upload(std::vector<float>(5)), GpuError);
    REQUIRE_THROWS_AS(tensor.reshape({2, 2}), GpuError);
}

TEST_CASE("GpuTensorView subview slices with correct offsets", "[tensor]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuTensor<float, 2> tensor(ctx, {3, 4}, GpuBuffer::Type::HostCoherent);
    std::vector<float> values(12);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(i);
    tensor.upload(values);

    auto view = tensor.view();
    auto sub = view.subview({1, 2}, {2, 2});
    REQUIRE(sub.shape() == std::array<size_t, 2>{2, 2});
    REQUIRE(sub.elements() == 4);
    REQUIRE(sub.descriptorInfo().offset == 6 * sizeof(float));

    auto toEdge = view.subview({1, 1});
    REQUIRE(toEdge.shape() == std::array<size_t, 2>{2, 3});
    REQUIRE(toEdge.descriptorInfo().offset == 5 * sizeof(float));
}

TEST_CASE("GpuTensorView subview validates bounds", "[tensor]")
{
    auto &ctx = GpuTestFixture::ctx();
    GpuTensor<float, 2> tensor(ctx, {3, 4}, GpuBuffer::Type::HostCoherent);
    auto view = tensor.view();

    REQUIRE_THROWS_AS(view.subview({4, 0}, {1, 1}), GpuError);
    REQUIRE_THROWS_AS(view.subview({0, 0}, {3, 5}), GpuError);
    REQUIRE_THROWS_AS(view.subview({1, 5}), GpuError);

    GpuTensorView<float, 2> empty;
    REQUIRE_THROWS_AS(empty.subview({0, 0}), GpuError);
}
