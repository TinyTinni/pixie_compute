# -------------------------------------------------------------------
# pixie_compute template — grab this directory and start implementing
# GPU compute ideas.
#
# Layout:
#   src/       host program + a reusable compute function
#   shaders/   .slang kernels, compiled to SPIR-V at runtime
#   tests/     Catch2 tests (off by default in this build)
#
# This project consumes pixie_compute two ways:
#   * default: fetch pixie_compute from GitHub with FetchContent
#              (zero setup, first build compiles slang from source)
#   * -DPIXIE_TEMPLATE_USE_INSTALLED=ON: find an installed pixie_compute
#              with find_package(). Point CMake at it with
#              -DCMAKE_PREFIX_PATH=/path/to/pixie-install
# -------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(pixie_compute_template LANGUAGES CXX)

option(PIXIE_TEMPLATE_USE_INSTALLED
       "Use an installed pixie_compute via find_package() instead of fetching from GitHub"
       OFF)

option(PIXIE_TEMPLATE_BUILD_TESTS "Build the Catch2 test suite" ON)

if(PIXIE_TEMPLATE_USE_INSTALLED)
    find_package(pixie_compute REQUIRED)
    set(PIXIE_TEMPLATE_LIB pixie_compute::pixie_compute)
else()
    include(FetchContent)
    # Do not build the library's own tests/examples when consumed as a subdirectory.
    set(PIXIE_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PIXIE_COMPUTE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        pixie_compute
        GIT_REPOSITORY https://github.com/TinyTinni/pixie_compute.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pixie_compute)
    # When consumed via add_subdirectory the pixie_compute::pixie_compute
    # alias is not visible here, so link the plain target name.
    set(PIXIE_TEMPLATE_LIB pixie_compute)
endif()

# ---- Core compute logic, shared by the app and the tests ------------------
add_library(pixie_template_core
    src/vector_add.cpp
)

target_link_libraries(pixie_template_core PUBLIC ${PIXIE_TEMPLATE_LIB})

target_include_directories(pixie_template_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Bake the absolute shader directory in so the .slang file is found at
# runtime regardless of the working directory.
target_compile_definitions(pixie_template_core PUBLIC
    PIXIE_TEMPLATE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

# ---- Application -----------------------------------------------------------
add_executable(compute_template src/main.cpp)
target_link_libraries(compute_template PRIVATE pixie_template_core)

# ---- Tests -----------------------------------------------------------------
if(PIXIE_TEMPLATE_BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.15.0
    )
    set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Catch2)

    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    include(Catch)

    add_subdirectory(tests)
endif()
