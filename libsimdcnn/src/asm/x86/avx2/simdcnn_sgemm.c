#include "simdcnn_sgemm.h"
#include <assert.h>
#include <omp.h>
#include <simdcnn/debug.h>
#include <simdcnn/errors.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// Since this is a static function, I removed it from the header.
static void simdcnn_sgemm_cleanup_avx2_(float **packed_As, float **packed_Bs, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        if (packed_As[i] != NULL)
            free(packed_As[i]);

        if (packed_Bs[i] != NULL)
            free(packed_Bs[i]);
    }

    if (packed_As != NULL)
        free(packed_As);

    if (packed_Bs != NULL)
        free(packed_Bs);
}

/// OpenBLIS-like SGEMM
/// C = aAB + bC
/// A, B and C are supposed to be row-major matrices of sizes M * K, K * N, and M * N
/// A, B and C SHOULD NOT overlap.
simdcnn_sgemm_error_t simdcnn_sgemm_avx2(float *const C, const float alpha, const float beta, const float *const A,
                                         const float *const B, const uint64_t M, const uint64_t K, const uint64_t N)
{
    const size_t size_A = (size_t)M * K;
    const size_t size_B = (size_t)K * N;
    const size_t size_C = (size_t)M * N;

    assert(!simdcnn_debug_overlaps_f(A, size_A, C, size_C));
    assert(!simdcnn_debug_overlaps_f(B, size_B, C, size_C));

#ifdef SIMDCNN_SGEMM_ERRORS_ENABLED
    if (simdcnn_debug_overlaps_f(A, size_A, C, size_C))
    {
        return SIMDCNN_SGEMM_MATRIX_ALIASING;
    }

    if (simdcnn_debug_overlaps_f(B, size_B, C, size_C))
    {
        return SIMDCNN_SGEMM_MATRIX_ALIASING;
    }
#endif

    size_t max_threads = omp_get_max_threads();
    size_t num_threads =
        (max_threads > SIMDCNN_MATMUL_AVX2_MAX_THREADS) ? SIMDCNN_MATMUL_AVX2_MAX_THREADS : max_threads;

    float **packed_As = (float **)malloc(num_threads * sizeof(float *));
    if (!packed_As)
    {
        return SIMDCNN_SGEMM_OUT_OF_MEMORY;
    }

    float **packed_Bs = (float **)malloc(num_threads * sizeof(float *));
    if (!packed_Bs)
    {
        free(packed_As);
        return SIMDCNN_SGEMM_OUT_OF_MEMORY;
    }

    const size_t packed_A_size = (SIMDCNN_MATMUL_AVX2_MC * SIMDCNN_MATMUL_AVX2_KC + (SIMDCNN_PAGE_ALIGNMENT - 1)) &
                                 ~(SIMDCNN_PAGE_ALIGNMENT - 1);

    const size_t packed_B_size = (SIMDCNN_MATMUL_AVX2_NC * SIMDCNN_MATMUL_AVX2_KC + (SIMDCNN_PAGE_ALIGNMENT - 1)) &
                                 ~(SIMDCNN_PAGE_ALIGNMENT - 1);

    for (size_t i = 0; i < num_threads; ++i)
    {
        packed_As[i] = (float *)aligned_alloc(SIMDCNN_PAGE_ALIGNMENT, packed_A_size * sizeof(float));

        if (!packed_As[i])
        {
            simdcnn_sgemm_cleanup_avx2_(packed_As, packed_Bs, i);
            return SIMDCNN_SGEMM_OUT_OF_MEMORY;
        }

        packed_Bs[i] = (float *)aligned_alloc(SIMDCNN_PAGE_ALIGNMENT, packed_B_size * sizeof(float));
        if (!packed_Bs[i])
        {
            free(packed_As[i]);
            simdcnn_sgemm_cleanup_avx2_(packed_As, packed_Bs, i);
            return SIMDCNN_SGEMM_OUT_OF_MEMORY;
        }
    }

#pragma omp parallel num_threads((int)num_threads)
    {
        size_t tid = omp_get_thread_num();
        float *packedA = packed_As[tid];
        float *packedB = packed_Bs[tid];

#pragma omp for schedule(dynamic)
        for (size_t ii = 0; ii < M; ii += SIMDCNN_MATMUL_AVX2_MR)
        {
            for (size_t kk = 0; kk < K; kk += SIMDCNN_MATMUL_AVX2_KC)
            {
            }
        }
    }

    simdcnn_sgemm_cleanup_avx2_(packed_As, packed_Bs, num_threads);

    return SIMDCNN_SGEMM_SUCCESS;
}