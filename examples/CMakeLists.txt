# These examples compile their shaders at runtime with the Slang-based
# GpuComputePipeline constructor, so they need PIXIE_COMPUTE_ENABLE_SLANG.
if(PIXIE_COMPUTE_ENABLE_SLANG)
    foreach(example vector_add monte_carlo_pi bayesian_mcmc reduce histogram pipeline pipeline_mt image_demo renderdoc_capture cooperative_matrix)
        add_executable(pixie_example_${example} ${example}.cpp)
        target_link_libraries(pixie_example_${example} PRIVATE pixie_compute::pixie_compute)
        set_target_properties(pixie_example_${example} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/$<CONFIG>)
        add_test(NAME pixie_example_${example} COMMAND pixie_example_${example})
        pixie_deploy_slang_dlls(pixie_example_${example})
    endforeach()
endif()

# Embedded-shaders example: the shader is compiled to SPIR-V at build time
# and embedded via pixie_add_shaders (cmake/PixieShaders.cmake). No runtime
# Slang compiler needed, so it builds even with PIXIE_COMPUTE_ENABLE_SLANG=OFF.
pixie_shaders_available(pixie_embedded_shaders_ok)
if(pixie_embedded_shaders_ok)
    add_executable(pixie_example_embedded_shaders embedded_shaders.cpp)
    pixie_add_shaders(pixie_example_embedded_shaders
        SHADERS shaders/vector_add.slang
        ENTRY_POINTS main
        OUTPUT_NAME embedded_shaders
    )
    target_link_libraries(pixie_example_embedded_shaders PRIVATE pixie_compute::pixie_compute)
    set_target_properties(pixie_example_embedded_shaders PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/$<CONFIG>)
    add_test(NAME pixie_example_embedded_shaders COMMAND pixie_example_embedded_shaders)
    pixie_deploy_slang_dlls(pixie_example_embedded_shaders)
endif()
