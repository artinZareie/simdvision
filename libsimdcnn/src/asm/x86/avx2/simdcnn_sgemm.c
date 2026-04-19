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
#pragma omp parallel for schedule(dynamic)
    for (uint64_t jc = 0; jc < N; jc += MATMUL_AVX2_NC)
    {
        const uint64_t end_jc = N < jc + MATMUL_AVX2_NC ? N : jc + MATMUL_AVX2_NC;
        // Pack B (copy selected column set from each row to be continguous)

        for (uint64_t pc = 0; pc < K; pc += MATMUL_AVX2_KC)
        {
            const uint64_t end_pc = K < pc + MATMUL_AVX2_KC ? K : pc + MATMUL_AVX2_KC;
            // Pack A (copy selected column set from each row to be continguous)

            for (uint64_t ic = 0; ic < M; ic += MATMUL_AVX2_MC)
            {
                const uint64_t end_ic = M < ic + MATMUL_AVX2_MC ? M : ic + MATMUL_AVX2_MC;

                for (uint64_t jr = jc; jr < end_jc; jr += MATMUL_AVX2_NR)
                {
                    const uint64_t end_jr = end_jc < jr + MATMUL_AVX2_NR ? end_jc : jr + MATMUL_AVX2_NR;

                    for (uint64_t ir = ic; ir < end_ic; ir += MATMUL_AVX2_MR)
                    {
                        const uint64_t end_ir = end_ic < ir + MATMUL_AVX2_MR ? end_ic : ir + MATMUL_AVX2_NR;

                        // Call ukernel
                    }
                }
            }
        }
    }
}