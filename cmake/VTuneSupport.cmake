if(NOT SIMD_CNN_VTUNE_SUPPORT)
    return()
endif()

#TODO: Please replace the version with your own
set(_VTUNE_HINT_DIRS
    "/opt/intel/oneapi/vtune/2025.0/sdk"
    "C:/Program Files (x86)/Intel/oneAPI/vtune/latest/sdk"
    $ENV{VTUNE_PROFILER_2024_DIR}/sdk
    $ENV{VTUNE_PROFILER_2025_DIR}/sdk
)

find_path(ITTNOTIFY_INCLUDE_DIR
    NAMES ittnotify.h
    PATH_SUFFIXES include
    HINTS ${_VTUNE_HINT_DIRS}
)

find_library(ITTNOTIFY_LIBRARY
    NAMES ittnotify
    PATH_SUFFIXES lib lib64
    HINTS ${_VTUNE_HINT_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ITTNOTIFY DEFAULT_MSG
    ITTNOTIFY_INCLUDE_DIR
    ITTNOTIFY_LIBRARY
)

if(ITTNOTIFY_FOUND)
    message(STATUS "VTune ITT Notify found:")
    message(STATUS "  include: ${ITTNOTIFY_INCLUDE_DIR}")
    message(STATUS "  library: ${ITTNOTIFY_LIBRARY}")

    set(ITTNOTIFY_INCLUDE_DIR ${ITTNOTIFY_INCLUDE_DIR})
    set(ITTNOTIFY_LIBRARY ${ITTNOTIFY_LIBRARY})

    set(SIMD_CNN_VTUNE_AVAILABLE ON)
else()
    message(WARNING "VTune ITT Notify NOT found; disabling SIMD_CNN_VTUNE_SUPPORT")
    set(SIMD_CNN_VTUNE_AVAILABLE OFF)
endif()
