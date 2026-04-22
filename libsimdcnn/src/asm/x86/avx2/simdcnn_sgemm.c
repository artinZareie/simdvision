#include <stdint.h>

#define MATMUL_AVX2_MC 128 /// Macro-tile size in M dimension
#define MATMUL_AVX2_NC 128 /// Macro-tile size in N dimension
#define MATMUL_AVX2_KC 256 /// Macro-tile size in K dimension

#define MATMUL_AVX2_MR 8  /// Micro-tile size in M dimension
#define MATMUL_AVX2_NR 12 /// Micro-tile size in N dimension

/// OpenBLIS-like SGEMM
/// C = aAB + C
void simdcnn_sgemm_avx2(float *dst, float alpha, float *A, float *B, uint64_t M, uint64_t N, uint64_t K)
{
}