#pragma once

/// Single-include convenience header for pixie_compute. Pulls in the whole
/// public API. Include only the specific headers you need if you prefer
/// faster compiles.

#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/gpu_image.hpp"
#include "pixie_compute/gpu_kernel.hpp"
#include "pixie_compute/gpu_profiler.hpp"
#include "pixie_compute/gpu_stream.hpp"
#include "pixie_compute/gpu_tensor.hpp"
#include "pixie_compute/gpu_timeline_semaphore.hpp"
#include "pixie_compute/gpu_timer.hpp"
#include "pixie_compute/gpu_vector.hpp"
#include "pixie_compute/renderdoc_capture.hpp"
#include "pixie_compute/shader_compiler.hpp"
#include "pixie_compute/spirv_loader.hpp"
#include "pixie_compute/utility.hpp"
