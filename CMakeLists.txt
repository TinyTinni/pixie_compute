# -------------------------------------------------------------------
# pixie_compute — lightweight Vulkan compute utility library
# -------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(pixie_compute VERSION 0.1.0 LANGUAGES CXX)

# ---- Dependencies via CPM ----
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Offline shader compilation (slangc -> SPIR-V -> embedded arrays) ----
include(PixieShaders)

# ---- Vulkan (via volk) ----
CPMAddPackage(
    NAME         volk
    GITHUB_REPOSITORY zeux/volk
    GIT_TAG      vulkan-sdk-1.4.350.0
    OPTIONS
        "VOLK_INSTALL ON"
)

# ---- Slang shading language (optional) ----
option(PIXIE_COMPUTE_ENABLE_SLANG "Enable runtime Slang shader compilation" ON)

if(PIXIE_COMPUTE_ENABLE_SLANG)
    find_package(Slang QUIET)

    if(Slang_FOUND)
        message(STATUS "Found system shader-slang: ${Slang_LIBRARY}")
    else()
        message(STATUS "shader-slang not found — building from source via CPM")
        CPMAddPackage(
            NAME         slang
            GITHUB_REPOSITORY shader-slang/slang
            GIT_TAG      v2026.8.1
            OPTIONS
                "SLANG_BUILD_TESTS OFF"
                "SLANG_BUILD_EXAMPLES OFF"
                "SLANG_ENABLE_GFX OFF"
                "SLANG_ENABLE_TESTS OFF"
                "SLANG_ENABLE_EXAMPLES OFF"
                "SLANG_ENABLE_SLANG_RHI OFF"
                "SLANG_ENABLE_SLANG_GLSLANG ON"
                "SLANG_ENABLE_DXIL OFF"
                "SLANG_ENABLE_CUDA OFF"
                "SLANG_ENABLE_OPTIX OFF"
                "SLANG_ENABLE_NVAPI OFF"
                "SLANG_ENABLE_AFTERMATH OFF"
                "SLANG_ENABLE_XLIB OFF"
                "SLANG_ENABLE_SLANGC ON"
                "SLANG_ENABLE_SLANGD OFF"
                "SLANG_ENABLE_SLANGI OFF"
                "SLANG_ENABLE_SLANGRT OFF"
                "SLANG_ENABLE_REPLAYER OFF"
                "SLANG_SLANG_LLVM_FLAVOR DISABLE"
                "SLANG_EXCLUDE_DAWN ON"
                "SLANG_EXCLUDE_TINT ON"
                "SLANG_ENABLE_SPLIT_DEBUG_INFO OFF"
        )
    endif()
endif()

# ---- Vulkan Memory Allocator ----
CPMAddPackage(
    NAME         VulkanMemoryAllocator
    GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG      v3.1.0
)

# ---- Library target ----
set(PIXIE_SOURCES
    src/gpu_allocator.cpp
    src/gpu_context.cpp
    src/gpu_buffer.cpp
    src/gpu_command_buffer.cpp
    src/gpu_compute_pipeline.cpp
    src/gpu_image.cpp
    src/gpu_profiler.cpp
    src/gpu_timer.cpp
    src/gpu_timeline_semaphore.cpp
    src/renderdoc_capture.cpp
    src/spirv_loader.cpp
)

if(PIXIE_COMPUTE_ENABLE_SLANG)
    list(APPEND PIXIE_SOURCES
        src/shader_compiler.cpp
    )
endif()

add_library(pixie_compute ${PIXIE_SOURCES})

add_library(pixie_compute::pixie_compute ALIAS pixie_compute)
target_compile_features(pixie_compute PUBLIC cxx_std_20)

target_include_directories(pixie_compute
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_include_directories(pixie_compute SYSTEM
    PRIVATE
        ${VulkanMemoryAllocator_SOURCE_DIR}/include
)

target_link_libraries(pixie_compute
    PUBLIC
        volk::volk
)

target_compile_definitions(pixie_compute PUBLIC VK_NO_PROTOTYPES)

if(PIXIE_COMPUTE_ENABLE_SLANG)
    target_compile_definitions(pixie_compute PUBLIC PIXIE_COMPUTE_HAS_SLANG)
endif()

# ---- Compiler warnings ----
if(MSVC)
    target_compile_options(pixie_compute PRIVATE /W4)
else()
    target_compile_options(pixie_compute PRIVATE
        -Wall -Wextra -Wconversion -Wsign-conversion)
endif()

if(PIXIE_COMPUTE_ENABLE_SLANG)
    if(TARGET slang::slang)
        target_link_libraries(pixie_compute PUBLIC slang::slang)
    elseif(TARGET slang)
        target_link_libraries(pixie_compute PUBLIC slang)
    endif()
endif()

# ---- Tests ----
option(PIXIE_COMPUTE_BUILD_TESTS "Build tests" ON)

if(PIXIE_COMPUTE_BUILD_TESTS)
    enable_testing()

    CPMAddPackage(
        NAME         Catch2
        GITHUB_REPOSITORY catchorg/Catch2
        GIT_TAG      v3.15.0
        EXCLUDE_FROM_ALL YES
    )
    list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")

    CPMAddPackage(
        NAME         rapidcheck
        GITHUB_REPOSITORY emil-e/rapidcheck
        GIT_TAG      b2d9ed2dddefc4b84318d664b4f221eb792d89c7
        EXCLUDE_FROM_ALL YES
    )

    add_subdirectory(tests)
endif()

# ---- Examples ----
option(PIXIE_COMPUTE_BUILD_EXAMPLES "Build examples" ON)

if(PIXIE_COMPUTE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---- Install & packaging ----
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS pixie_compute
    EXPORT pixie_computeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/pixie_compute
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT pixie_computeTargets
    FILE pixie_computeTargets.cmake
    NAMESPACE pixie_compute::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pixie_compute
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pixie_computeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pixie_computeConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pixie_compute
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/pixie_computeConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/pixie_computeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/pixie_computeConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindSlang.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pixie_compute
)
