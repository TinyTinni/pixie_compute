# FindSlang.cmake — Find shader-slang (not S-Lang).
#
# Tries in order:
#   1. CONFIG mode for properly CMake-installed packages (vcpkg, self-built)
#   2. pkg-config
#   3. Manual search: /opt/shader-slang-bin (AUR shader-slang-bin),
#      ${Slang_ROOT}, ${CMAKE_PREFIX_PATH}, /usr, /usr/local
#
# Targets:
#   slang::slang  — imported target (created with this name)
#
# Variables:
#   Slang_FOUND
#   Slang_INCLUDE_DIR
#   Slang_LIBRARY

# Try CONFIG mode first — covers vcpkg, self-built with install
find_package(Slang CONFIG QUIET)
if(Slang_FOUND)
  return()
endif()

# Try pkg-config
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_Slang QUIET shader-slang)
endif()

# Build hints from CMAKE_PREFIX_PATH
set(_slang_include_hints)
set(_slang_lib_hints)
foreach(_prefix ${CMAKE_PREFIX_PATH})
  list(APPEND _slang_include_hints "${_prefix}/include/shader-slang")
  list(APPEND _slang_lib_hints "${_prefix}/lib")
endforeach()

find_path(Slang_INCLUDE_DIR
  NAMES slang.h
  PATHS
    /opt/shader-slang-bin/include/shader-slang
    /opt/shader-slang-bin/include
    /usr/include/shader-slang
    /usr/include
    /usr/local/include/shader-slang
    /usr/local/include
    ${PC_Slang_INCLUDEDIR}
    ${Slang_ROOT}/include/shader-slang
    ${Slang_ROOT}/include
    ${_slang_include_hints}
  NO_DEFAULT_PATH
)

find_library(Slang_LIBRARY
  NAMES slang shader-slang
  PATHS
    /opt/shader-slang-bin/lib
    ${PC_Slang_LIBDIR}
    ${Slang_ROOT}/lib
    ${_slang_lib_hints}
  NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Slang
  REQUIRED_VARS Slang_LIBRARY Slang_INCLUDE_DIR
)

if(Slang_FOUND AND NOT TARGET slang::slang)
  add_library(slang::slang UNKNOWN IMPORTED)
  set_target_properties(slang::slang PROPERTIES
    IMPORTED_LOCATION "${Slang_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Slang_INCLUDE_DIR}"
  )

endif()

mark_as_advanced(Slang_INCLUDE_DIR Slang_LIBRARY)
