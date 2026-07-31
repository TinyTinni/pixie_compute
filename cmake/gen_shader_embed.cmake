# gen_shader_embed.cmake — generate C++ source/header embedding SPIR-V binaries.
#
# Required -D args:
#   SPV_FILES  ; SPIR-V file paths
#   SYMBOLS    ; matching C++ symbol names
#   OUT_CPP    ; generated .cpp path
#   OUT_HPP    ; generated .hpp path
#
# Emits, per symbol (namespace pix::shaders):
#   extern const uint32_t <symbol>[];
#   extern const uint32_t <symbol>_count;

if(NOT DEFINED SPV_FILES OR NOT DEFINED SYMBOLS OR NOT DEFINED OUT_CPP OR NOT DEFINED OUT_HPP)
    message(FATAL_ERROR "gen_shader_embed.cmake requires SPV_FILES, SYMBOLS, OUT_CPP, OUT_HPP")
endif()

list(LENGTH SPV_FILES _n_files)
list(LENGTH SYMBOLS _n_symbols)
if(NOT _n_files EQUAL _n_symbols)
    message(FATAL_ERROR "gen_shader_embed: SPV_FILES/SYMBOLS length mismatch")
endif()

set(_decl_body)
set(_array_body)

foreach(_i RANGE 0 ${_n_files})
    if(_i EQUAL _n_files)
        break()
    endif()
    list(GET SPV_FILES ${_i} _spv)
    list(GET SYMBOLS ${_i} _symbol)

    file(READ "${_spv}" _hex HEX)
    string(LENGTH "${_hex}" _hex_len)
    math(EXPR _byte_count "${_hex_len} / 2")
    math(EXPR _word_count "${_byte_count} / 4")
    if(NOT _word_count GREATER 0)
        message(FATAL_ERROR "gen_shader_embed: empty SPIR-V output from ${_spv}")
    endif()
    math(EXPR _last "${_word_count} - 1")

    set(_lines)
    set(_buf "")
    foreach(_w RANGE 0 ${_word_count})
        if(_w EQUAL _word_count)
            break()
        endif()
        math(EXPR _pos "${_w} * 8")
        string(SUBSTRING "${_hex}" ${_pos} 8 _group)
        # Little-endian word: bytes stored reversed (b3 b2 b1 b0)
        string(SUBSTRING "${_group}" 6 2 _b0)
        string(SUBSTRING "${_group}" 4 2 _b1)
        string(SUBSTRING "${_group}" 2 2 _b2)
        string(SUBSTRING "${_group}" 0 2 _b3)
        string(CONCAT _word "0x${_b0}${_b1}${_b2}${_b3},")
        string(APPEND _buf "${_word} ")
        math(EXPR _rem "${_w} % 4")
        if(_rem EQUAL 3 OR _w EQUAL _last)
            string(APPEND _lines "    ${_buf}\n")
            set(_buf "")
        endif()
    endforeach()

    string(APPEND _decl_body "extern const uint32_t ${_symbol}[];\nextern const uint32_t ${_symbol}_count;\n")
    string(APPEND _array_body "extern const uint32_t ${_symbol}[] = {\n${_lines}};\nextern const uint32_t ${_symbol}_count = sizeof(${_symbol}) / sizeof(${_symbol}[0]);\n\n")
endforeach()

set(_hdr_content "
#pragma once

#include <cstdint>

namespace pix
{
namespace shaders
{

${_decl_body}
} // namespace shaders
} // namespace pix
")

set(_cpp_content "
#include <cstdint>

namespace pix
{
namespace shaders
{

${_array_body}
} // namespace shaders
} // namespace pix
")

file(WRITE "${OUT_HPP}" "${_hdr_content}")
file(WRITE "${OUT_CPP}" "${_cpp_content}")
