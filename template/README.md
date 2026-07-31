# pixie_compute template

A minimal, self-contained starting point for GPU compute experiments with
[pixie_compute](https://github.com/TinyTinni/pixie_compute). Copy this
directory wherever you like and start editing.

```
src/main.cpp           host program (entry point, timing, verification)
src/vector_add.hpp/.cpp a reusable compute function (shared with the tests)
shaders/vector_add.slang  the GPU kernel, compiled to SPIR-V at runtime
tests/                 Catch2 test suite
```

It does one thing end-to-end: vector add on the GPU (allocate → upload →
dispatch → read back → verify), with each step marked and commented so you can
swap in your own kernel.

## Prerequisites

- CMake ≥ 3.20 and a C++20 compiler
- A Vulkan driver + the Vulkan SDK headers
- Slang is fetched and built automatically (only the first build is slow)

## Build and run

By default the template fetches pixie_compute from GitHub:

```sh
cmake -B build
cmake --build build -j
./build/compute_template
```

The first configure compiles the Slang shader compiler from source; subsequent
builds are fast.

### Use an installed pixie_compute instead

Install pixie_compute once (`cmake --install`), then point the template at it:

```sh
cmake -B build -DPIXIE_TEMPLATE_USE_INSTALLED=ON -DCMAKE_PREFIX_PATH=/path/to/pixie-install
cmake --build build -j
./build/compute_template
```

## Run the tests

The template ships with a Catch2 suite covering both the shader compilation and
the full GPU pipeline (GPU tests skip automatically when Vulkan is missing):

```sh
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## What to change

1. **The kernel** — replace the buffers/body in `shaders/vector_add.slang`.
   Slang is an HLSL-style language: buffers are `RWStructuredBuffer<T>`, the
   launch size is `[numthreads(x,y,z)]`, and `SV_DispatchThreadID` is the global
   thread index.
2. **The data** — adjust `N` in `src/main.cpp` and how `hostA`/`hostB` are
   filled. Match the shader's element type to the `float` in the buffers.
3. **Buffer bindings** — the order of `descriptorInfo()` entries in the
   `GpuComputePipeline` constructor must match the `register(uN)` slots.
4. **Workgroup size** — the `[numthreads]` in the shader and the
   `workgroupSize` argument to the pipeline must agree.
5. **New kernels** — drop a new `.slang` file in `shaders/` and call
   `compileSlangModule("name", "main", {PIXIE_TEMPLATE_SHADER_DIR})` to get
   SPIR-V for a `GpuComputePipeline`.

## API cheat sheet

| What you need                     | Use                                             |
| --------------------------------- | ----------------------------------------------- |
| One context per process           | `GpuContext ctx;`                               |
| CPU-visible GPU memory            | `GpuBuffer b(ctx, size, GpuBuffer::Type::HostCoherent);` |
| GPU-only memory (faster, staged)  | `GpuBuffer::Type::Device`                       |
| Copy data to/from GPU             | `b.upload(span); b.download(span);`             |
| Compile a .slang file to SPIR-V   | `compileSlangModule(name, entry, {shaderDir})`  |
| Compile shader + build pipeline   | `GpuComputePipeline p(ctx, spirv, bindings, pcSize, wgX, wgY, wgZ);` |
| Run it once, block until done     | `oneShotDispatch(ctx, p, groupsX, groupsY, groupsZ);` |
| Round up threads to workgroups    | `workgroupCount(N, wgSize)`                     |
| Errors                            | `catch (const pix::GpuError &e)`                |
| Device summary                    | `ctx.infoDump()`                                |

For more: multi-pass/async (`GpuCommandBuffer`, `GpuTimelineSemaphore`),
timing (`GpuTimer`, `GpuProfiler`), images (`GpuImage`), and offline-embedded
shaders (`pixie_add_shaders`) are covered by the examples in the
[pixie_compute repository](https://github.com/TinyTinni/pixie_compute/tree/master/examples).
