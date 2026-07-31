#pragma once

#include "pixie_compute/gpu_context.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace pix
{

/// Run the vector-add kernel on the GPU: returns c where c[i] = a[i] + b[i].
/// Compiles shaders/vector_add.slang at runtime, so the shader directory must
/// be available next to the binary (baked in by CMake as PIXIE_TEMPLATE_SHADER_DIR).
/// Throws GpuError on mismatched/empty input or any GPU failure.
std::vector<float> vectorAdd(GpuContext &ctx, std::span<const float> a, std::span<const float> b);

} // namespace pix
