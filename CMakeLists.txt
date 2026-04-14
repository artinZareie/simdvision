cmake_minimum_required(VERSION 3.22)
project(simd_cnn LANGUAGES C ASM_NASM)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ArchitectureSupportCheck)
include(NasmSupport)

option(SIMD_CNN_BUILD_TESTS "Build tests for libsimdcnn" ON)

add_subdirectory(libsimdcnn)
add_subdirectory(simdcnn-cli)

if (SIMD_CNN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(libsimdcnn/tests)
endif()
