cmake_minimum_required(VERSION 3.22)
project(simd_cnn LANGUAGES C ASM_NASM)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(SIMD_CNN_BUILD_TESTS "Build tests for libsimdcnn" ON)
option(SIMD_CNN_BUILD_BENCHMARK "Build benchmarks for libsimdcnn" ON)
option(SIMD_CNN_VTUNE_SUPPORT "Include VTune headers to benchmarking and enable VTune ITT instrumentation" ON)

include(ArchitectureSupportCheck)
include(NasmSupport)
include(VTuneSupport)

add_subdirectory(libsimdcnn)
add_subdirectory(simdcnn-cli)

if(SIMD_CNN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(libsimdcnn/tests)
endif()

if(SIMD_CNN_BUILD_BENCHMARK)
    add_subdirectory(libsimdcnn/benchmark)
endif()
