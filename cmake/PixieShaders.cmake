# -------------------------------------------------------------------
# PixieShaders.cmake — build-time Slang -> SPIR-V compilation.
#
# Compiles .slang shaders to SPIR-V with slangc during the build and
# embeds them in the target as C++ arrays (namespace pix::shaders),
# so the runtime needs no shader compiler.
# -------------------------------------------------------------------

option(PIXIE_COMPUTE_ENABLE_OFFLINE_SHADERS "Compile .slang shaders to SPIR-V at build time" ON)

# Locate slangc: PATH first, then next to a known Slang library, then a
# slangc target from a Slang build added via add_subdirectory.
find_program(PIXIE_SLANGC_EXECUTABLE slangc)
if(NOT PIXIE_SLANGC_EXECUTABLE AND Slang_LIBRARY)
    get_filename_component(_pixie_slang_bin "${Slang_LIBRARY}" DIRECTORY)
    get_filename_component(_pixie_slang_prefix "${_pixie_slang_bin}" DIRECTORY)
    find_program(PIXIE_SLANGC_EXECUTABLE slangc HINTS "${_pixie_slang_prefix}/bin")
endif()

# Returns ON/OFF in <result> depending on whether the offline shader
# pipeline is enabled and a slangc is available.
function(pixie_shaders_available result)
    if(NOT PIXIE_COMPUTE_ENABLE_OFFLINE_SHADERS)
        set(${result} OFF PARENT_SCOPE)
        return()
    endif()
    if(PIXIE_SLANGC_EXECUTABLE OR TARGET slangc)
        set(${result} ON PARENT_SCOPE)
    else()
        set(${result} OFF PARENT_SCOPE)
    endif()
endfunction()

# pixie_add_shaders(<target>
#     SHADERS <shader.slang>...
#     [ENTRY_POINTS <entry>...]   # one entry for all shaders, or one per shader
#     [STAGE <compute|fragment|...>]  # default: compute
#     [SPIRV_VERSION <1.0|1.3|1.4|...>])  # default: 1.3 (Vulkan 1.1 baseline);
#                                         #   mapped to a Slang profile (cs_5_1/cs_6_0/cs_6_3)
#     [OUTPUT_NAME <name>])      # generated files <name>_shaders.hpp/.cpp (default: <target>)
#
# Compiles each shader to SPIR-V and generates <out_dir>/<name>_shaders.hpp/.cpp
# declaring, per (shader, entry):
#   namespace pix::shaders { extern const uint32_t <symbol>[]; extern const uint32_t <symbol>_count; }
# The symbol is the shader file stem (sanitized); non-main entries append _<entry>.
# The generated .cpp is added to <target> and the output dir to its include path.
function(pixie_add_shaders target)
    cmake_parse_arguments(ARGS "" "STAGE;SPIRV_VERSION;OUTPUT_NAME" "SHADERS;ENTRY_POINTS" ${ARGN})
    if(NOT ARGS_SHADERS)
        message(FATAL_ERROR "pixie_add_shaders: no SHADERS given")
    endif()
    if(NOT ARGS_STAGE)
        set(ARGS_STAGE compute)
    endif()
    if(NOT ARGS_SPIRV_VERSION)
        set(ARGS_SPIRV_VERSION 1.3)
    endif()
    if(NOT ARGS_OUTPUT_NAME)
        set(ARGS_OUTPUT_NAME "${target}")
    endif()

    pixie_shaders_available(_pixie_shaders_ok)
    if(NOT _pixie_shaders_ok)
        message(WARNING "pixie_add_shaders: offline shaders unavailable (slangc not found); skipping ${target}")
        return()
    endif()

    set(_slangc "${PIXIE_SLANGC_EXECUTABLE}")
    if(NOT _slangc)
        set(_slangc "$<TARGET_FILE:slangc>")
    endif()

    if(NOT ARGS_ENTRY_POINTS)
        set(ARGS_ENTRY_POINTS main)
    endif()
    list(LENGTH ARGS_SHADERS _shader_count)
    list(LENGTH ARGS_ENTRY_POINTS _entry_count)
    if(NOT _entry_count EQUAL 1 AND NOT _entry_count EQUAL _shader_count)
        message(FATAL_ERROR
            "pixie_add_shaders: ENTRY_POINTS must have 1 entry or one per shader")
    endif()

    set(_out_dir "${CMAKE_CURRENT_BINARY_DIR}/pixie_shaders")
    file(MAKE_DIRECTORY "${_out_dir}")

    set(_spv_files)
    set(_symbols)
    set(_shader_dependencies)
    set_property(GLOBAL PROPERTY PIXIE_SHADER_DEPENDENCY_VISITED "")
    set(_index 0)
    foreach(_src IN LISTS ARGS_SHADERS)
        get_filename_component(_src_abs "${_src}" ABSOLUTE)
        _pixie_collect_shader_dependencies("${_src_abs}" _src_deps)
        list(APPEND _shader_dependencies "${_src_abs}" ${_src_deps})
        get_filename_component(_stem "${_src_abs}" NAME_WE)
        string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" _symbol "${_stem}")
        if(_entry_count GREATER 1)
            list(GET ARGS_ENTRY_POINTS ${_index} _entry)
        else()
            set(_entry "${ARGS_ENTRY_POINTS}")
        endif()
        if(NOT _entry STREQUAL "main")
            set(_symbol "${_symbol}_${_entry}")
        endif()

        if(ARGS_SPIRV_VERSION VERSION_LESS "1.3")
            set(_profile "cs_5_1")  # emits SPIR-V 1.0
        elseif(ARGS_SPIRV_VERSION VERSION_LESS "1.4")
            set(_profile "cs_6_0")  # emits SPIR-V 1.3 (Vulkan 1.1 baseline)
        else()
            set(_profile "cs_6_3")  # emits SPIR-V 1.4
        endif()
        set(_spv "${_out_dir}/${_symbol}.spv")
        # Slang infers compute stage from [numthreads]. Passing -stage again
        # produces a duplicate-stage warning, but retain the option for shaders
        # whose stage cannot be inferred from their source.
        file(READ "${_src_abs}" _shader_source)
        set(_stage_args)
        if(NOT _shader_source MATCHES "numthreads[ \t]*\\(")
            list(APPEND _stage_args -stage "${ARGS_STAGE}")
        endif()
        add_custom_command(
            OUTPUT "${_spv}"
            COMMAND ${_slangc} "${_src_abs}" -target spirv ${_stage_args}
                    -entry "${_entry}" -profile "${_profile}" -o "${_spv}"
            DEPENDS ${_src_deps} "${_src_abs}" "${_slangc}"
            COMMENT "slangc: ${_src}"
            VERBATIM
        )
        list(APPEND _spv_files "${_spv}")
        list(APPEND _symbols "${_symbol}")
        math(EXPR _index "${_index} + 1")
    endforeach()

    list(REMOVE_DUPLICATES _shader_dependencies)
    set(_gen_cpp "${_out_dir}/${ARGS_OUTPUT_NAME}_shaders.cpp")
    set(_gen_hpp "${_out_dir}/${ARGS_OUTPUT_NAME}_shaders.hpp")
    add_custom_command(
        OUTPUT "${_gen_cpp}" "${_gen_hpp}"
        COMMAND ${CMAKE_COMMAND}
            "-DSPV_FILES=${_spv_files}"
            "-DSYMBOLS=${_symbols}"
            "-DOUT_CPP=${_gen_cpp}"
            "-DOUT_HPP=${_gen_hpp}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/gen_shader_embed.cmake"
        DEPENDS ${_spv_files} ${_shader_dependencies} "${_slangc}"
                "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/gen_shader_embed.cmake"
        COMMENT "Generating embedded shader arrays for ${target}"
        VERBATIM
    )

    target_sources(${target} PRIVATE "${_gen_cpp}" "${_gen_hpp}")
    target_include_directories(${target} PRIVATE "${_out_dir}")
endfunction()

# Collect local Slang includes recursively. Slang include search paths are not
# exposed by this helper, so resolve quoted/angle includes relative to the
# including file and track files that exist there.
function(_pixie_collect_shader_dependencies source_file result)
    get_property(_visited GLOBAL PROPERTY PIXIE_SHADER_DEPENDENCY_VISITED)
    if(NOT _visited)
        set(_visited)
    endif()
    list(FIND _visited "${source_file}" _seen)
    if(NOT _seen EQUAL -1)
        set(${result} "" PARENT_SCOPE)
        return()
    endif()
    list(APPEND _visited "${source_file}")
    set_property(GLOBAL PROPERTY PIXIE_SHADER_DEPENDENCY_VISITED "${_visited}")

    set(_deps)
    file(STRINGS "${source_file}" _lines REGEX "^[ \\t]*#[ \\t]*include[ \\t]+[\"<][^\">]+[\">]")
    get_filename_component(_source_dir "${source_file}" DIRECTORY)
    foreach(_line IN LISTS _lines)
        string(REGEX MATCH "#[ \\t]*include[ \\t]+[\"<]([^\">]+)[\">]" _match "${_line}")
        if(NOT _match)
            continue()
        endif()
        set(_include "${CMAKE_MATCH_1}")
        get_filename_component(_include_abs "${_source_dir}/${_include}" ABSOLUTE)
        if(EXISTS "${_include_abs}")
            list(APPEND _deps "${_include_abs}")
            _pixie_collect_shader_dependencies("${_include_abs}" _nested)
            list(APPEND _deps ${_nested})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _deps)
    set(${result} "${_deps}" PARENT_SCOPE)
endfunction()
