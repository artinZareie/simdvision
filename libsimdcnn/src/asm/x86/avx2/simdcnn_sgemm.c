#include <stdint.h>

#define MATMUL_AVX2_MC 128      /// Macro-tile size in M dimension
#define MATMUL_AVX2_NC 4096     /// Macro-tile size in N dimension
#define MATMUL_AVX2_KC 256      /// Macro-tile size in K dimension

void simdcnn_asm_sgemm_avx2(float *dst, float alpha, float *A, float *B, uint64_t M, uint64_t N, uint64_t P)
{
}