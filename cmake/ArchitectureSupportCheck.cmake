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