add_library(simdcnn SHARED)

include(CheckCSourceCompiles)

# Check if AVX2 is available
set(CMAKE_REQUIRED_FLAGS "-mavx2")
check_c_source_compiles("
    #include <immintrin.h>
    int main() {
        __m256 x = _mm256_set1_ps(1.0f);
        return (int)_mm256_cvtss_f32(x);
    }
" SIMD_CNN_AVX2_SUPPORTED)

# Check if SSE2 is available
set(CMAKE_REQUIRED_FLAGS "-msse2")
check_c_source_compiles("
    #include <emmintrin.h>
    int main() {
        __m128i x = _mm_set1_epi32(1);
        return (int)_mm_cvtsi128_si32(x);
    }
" SIMD_CNN_SSE2_SUPPORTED)

unset(CMAKE_REQUIRED_FLAGS)

if(SIMD_CNN_AVX2_SUPPORTED)
    message(STATUS "SIMD: using AVX2 kernels")
    set(SIMD_CNN_USE_AVX2 ON)
elseif(SIMD_CNN_SSE2_SUPPORTED)
    message(STATUS "SIMD: AVX2 not available, using SSE2 kernels")
    set(SIMD_CNN_USE_SSE2 ON)
else()
    message(FATAL_ERROR "Neither AVX2 nor SSE2 is supported by the compiler. Aborting.")
endif()

set(ASM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/asm)

if(SIMD_CNN_USE_AVX2)
    set(SIMD_KERNELS_LOCATION x86/avx2)
    set(EXTRA_ASM_FLAGS "-DARCH_AVX2=1")
elseif(SIMD_CNN_USE_SSE2)
    set(SIMD_KERNELS_LOCATION x86/sse2)
    set(EXTRA_ASM_FLAGS "-DARCH_SSE2=1")
endif()

set(LIBSIMDCNN_C_SOURCES
    src/simdcnn.c
)

set(LIBSIMDCNN_ASM_SOURCES
)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asm
)

target_compile_options(simdcnn PRIVATE
    -Wall -Wextra -pedantic
)

set_target_properties(simdcnn PROPERTIES VERSION 0.1.0 SOVERSION 0)
