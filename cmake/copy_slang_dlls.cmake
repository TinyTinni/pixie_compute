# Copy every *.dll from a source directory into a destination directory.
#
# Used at build time to deploy shared-library runtimes (e.g. Slang's slang.dll
# and its sibling downstream-compiler modules) next to executables that link
# them. Windows only searches the executable directory, system directories, and
# PATH, so relying on a build-tree or vcpkg location does not work when running
# from Visual Studio or CTest.
#
# Usage (from a POST_BUILD custom command):
#   cmake -P copy_slang_dlls.cmake <src_dir> <dst_dir>
if(NOT CMAKE_ARGV0 OR NOT CMAKE_ARGV1)
    message(FATAL_ERROR "usage: cmake -P copy_slang_dlls.cmake <src_dir> <dst_dir>")
endif()

set(_src_dir "${CMAKE_ARGV0}")
set(_dst_dir "${CMAKE_ARGV1}")

file(GLOB _dlls "${_src_dir}/*.dll")
foreach(_dll IN LISTS _dlls)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_dll}" "${_dst_dir}"
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        message(WARNING "copy_slang_dlls: failed to copy ${_dll}")
    endif()
endforeach()
