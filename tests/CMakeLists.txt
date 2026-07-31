list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)

add_executable(pixie_compute_tests
    test_shader_compiler.cpp
    test_gpu_buffer.cpp
    test_gpu_command_buffer.cpp
    test_gpu_compute_pipeline.cpp
    test_gpu_context_validation.cpp
    test_gpu_image.cpp
    test_gpu_profiler.cpp
    test_gpu_timer.cpp
    test_spirv_loader.cpp
    test_compile_slang_module.cpp
    test_debug_utils.cpp
    test_info_dump.cpp
    test_extensions.cpp
    test_memory_stats.cpp
    test_renderdoc_capture.cpp
    test_error_paths.cpp
    test_edge_cases.cpp
)

target_link_libraries(pixie_compute_tests
    PRIVATE
        pixie_compute::pixie_compute
        Catch2::Catch2WithMain
        rapidcheck
)

target_include_directories(pixie_compute_tests
    SYSTEM PRIVATE ${rapidcheck_SOURCE_DIR}/include
    SYSTEM PRIVATE ${rapidcheck_SOURCE_DIR}/extras/catch/include
)

# ---- Compiler warnings (tests) ----
if(MSVC)
    target_compile_options(pixie_compute_tests PRIVATE /W4)
else()
    target_compile_options(pixie_compute_tests PRIVATE
        -Wall -Wextra -Wconversion -Wsign-conversion)
endif()

# ---- Sanitizers (test builds only) ----
if(NOT MSVC AND NOT WIN32)
    target_compile_options(pixie_compute_tests PRIVATE -fsanitize=address,undefined)
    target_link_options(pixie_compute_tests PRIVATE -fsanitize=address,undefined)
elseif(MSVC)
    target_compile_options(pixie_compute_tests PRIVATE /fsanitize=address)
endif()

# ---- Blur engine tests (desktop only) ----
if(NOT ANDROID AND false) # do not add android examples yet
    add_executable(blur_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/android/app/cpp/test_frame_ring_buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/android/app/cpp/test_blur_engine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/android/app/cpp/frame_ring_buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/android/app/cpp/blur_engine.cpp
    )

    target_include_directories(blur_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../examples/android/app/cpp
    )

    target_link_libraries(blur_tests
        PRIVATE
            pixie_compute::pixie_compute
            Catch2::Catch2WithMain
    )

    if(MSVC)
        target_compile_options(blur_tests PRIVATE /W4)
    else()
        target_compile_options(blur_tests PRIVATE
            -Wall -Wextra -Wconversion -Wsign-conversion)
    endif()

    catch_discover_tests(blur_tests
        PROPERTIES
            ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan_suppressions.txt:detect_leaks=0"
    )
endif()

# LSAN_SUPPRESSIONS suppresses symbolized leaks; detect_leaks=0 disables leak reporting
# for Slang's runtime global init leak (dlopen'd, symbols unresolvable by LSAN).
catch_discover_tests(pixie_compute_tests
    PROPERTIES
        ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan_suppressions.txt:detect_leaks=0"
)
