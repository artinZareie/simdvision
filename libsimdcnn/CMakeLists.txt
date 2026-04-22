add_library(simdcnn SHARED)

find_package(OpenMP REQUIRED)
target_link_libraries(simdcnn PRIVATE OpenMP::OpenMP_C)

set(LIBSIMDCNN_ASM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/asm)

set(LIBSIMDCNN_C_SOURCES
    src/simdcnn.c
)

set(LIBSIMDCNN_ASM_SOURCES "")

if(SIMD_CNN_AVX2_SUPPORTED)
    message(STATUS "SIMD: Compiler supports AVX2, including AVX2 kernels")
    list(APPEND LIBSIMDCNN_ASM_SOURCES

        # vecadd
        ${LIBSIMDCNN_ASM_DIR}/x86/avx2/simdcnn_vecadd_f32.asm
        ${LIBSIMDCNN_ASM_DIR}/x86/avx2/simdcnn_vecadd_f64.asm

        # relu
        ${LIBSIMDCNN_ASM_DIR}/x86/avx2/simdcnn_relu_f32.asm
        ${LIBSIMDCNN_ASM_DIR}/x86/avx2/simdcnn_relu_f64.asm
    )

    # Define macro for C code
    target_compile_definitions(simdcnn PRIVATE HAVE_AVX2=1)
endif()

if(SIMD_CNN_SSE2_SUPPORTED)
    message(STATUS "SIMD: Compiler supports SSE2, including SSE2 kernels")

    # list(APPEND LIBSIMDCNN_ASM_SOURCES ${LIBSIMDCNN_ASM_DIR}/x86/sse2/simdcnn_vecadd_f32.asm)

    # Define macro fo C
    target_compile_definitions(simdcnn PRIVATE HAVE_SSE2=1)
endif()

if(NOT SIMD_CNN_AVX2_SUPPORTED AND NOT SIMD_CNN_SSE2_SUPPORTED)
    message(WARNING "Neither AVX2 nor SSE2 is supported by the compiler. Falling back to scalar C implementations.")
endif()

target_sources(simdcnn
    PRIVATE
    ${LIBSIMDCNN_C_SOURCES}
    ${LIBSIMDCNN_ASM_SOURCES}
)

target_include_directories(simdcnn
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(simdcnn
    PRIVATE
    ${LIBSIMDCNN_ASM_DIR}
)

target_compile_options(simdcnn PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-pedantic>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-pedantic>
    $<$<AND:$<COMPILE_LANGUAGE:ASM_NASM>,$<CONFIG:Debug>>:-g;-F;dwarf>
)

# Aligned-stack for x86inc.asm
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(simdcnn PRIVATE "$<$<COMPILE_LANGUAGE:ASM_NASM>:HAVE_ALIGNED_STACK=0>")
else()
    target_compile_definitions(simdcnn PRIVATE "$<$<COMPILE_LANGUAGE:ASM_NASM>:HAVE_ALIGNED_STACK=1>")
endif()

set_target_properties(simdcnn PROPERTIES VERSION 0.1.0 SOVERSION 0)
