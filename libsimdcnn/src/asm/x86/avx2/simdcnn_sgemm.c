#include <assert.h>
#include <immintrin.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <simdcnn/debug.h>
#include <simdcnn/def.h>
#include <simdcnn/errors.h>
#include <simdcnn/simdcnn_sgemm.h>

extern void simdcnn_sgemm_ukernel_6x16_avx2(float *c, uint64_t N, const float *a, const float *b, uint64_t size_kk);

/// Cleans up memory allocated for simdcnn_sgemm_avx2
static void simdcnn_sgemm_cleanup_avx2_(float **packed_As, float **packed_Bs, size_t n, bool free_A, bool free_B)
{
    for (size_t i = 0; i < n; ++i)
    {
        if (packed_As[i] != NULL && free_A)
            free(packed_As[i]);

        if (packed_Bs[i] != NULL && free_B)
            free(packed_Bs[i]);
    }

    if (packed_As != NULL)
        free(packed_As);

    if (packed_Bs != NULL)
        free(packed_Bs);
}

/// Copies a panel of A into the cache.
static void simdcnn_sgemm_pack_A_avx2_(float *packedA, const float *A, size_t K, size_t ii, size_t kk, size_t size_ii,
                                       size_t size_kk)
{
    for (size_t j = 0; j < SIMDCNN_SGEMM_AVX2_KC; ++j)
    {
#pragma GCC unroll 8
        for (size_t i = 0; i < SIMDCNN_SGEMM_AVX2_MC; ++i)
        {
            if (i >= size_ii || j >= size_kk)
            {
                packedA[i + j * SIMDCNN_SGEMM_AVX2_MC] = 0.0f;
            }
            else
            {
                packedA[i + j * SIMDCNN_SGEMM_AVX2_MC] = A[(i + ii) * K + (j + kk)];
            }
        }
    }
}

static void simdcnn_sgemm_pack_B_avx2_(float *packedB, const float *B, size_t N, size_t kk, size_t jj, size_t size_kk,
                                       size_t size_jj, float alpha)
{
    for (size_t i = 0; i < SIMDCNN_SGEMM_AVX2_KC; ++i)
    {
#pragma GCC unroll 8
        for (size_t j = 0; j < SIMDCNN_SGEMM_AVX2_NC; ++j)
        {
            if (i >= size_kk || j >= size_jj)
            {
                packedB[j + i * SIMDCNN_SGEMM_AVX2_NC] = 0.0f;
            }
            else
            {
                packedB[j + i * SIMDCNN_SGEMM_AVX2_NC] = B[(i + kk) * N + j + jj] * alpha;
            }
        }
    }
}

/// See header for fulll documentation.
simdcnn_sgemm_error_t simdcnn_sgemm_avx2(float *restrict C, float alpha, float beta, const float *A, const float *B,
                                         uint64_t M, uint64_t K, uint64_t N, float *packed_A_bun, float *packed_B_bun)
{
    const size_t size_A = (size_t)M * K;
    const size_t size_B = (size_t)K * N;
    const size_t size_C = (size_t)M * N;

    bool free_A = packed_A_bun == NULL;
    bool free_B = packed_B_bun == NULL;

    assert(!simdcnn_debug_overlaps_f(A, size_A, C, size_C));
    assert(!simdcnn_debug_overlaps_f(B, size_B, C, size_C));

    assert(!((uint64_t)packed_A_bun % SIMDCNN_PAGE_ALIGNMENT));
    assert(!((uint64_t)packed_B_bun % SIMDCNN_PAGE_ALIGNMENT));

#ifdef SIMDCNN_SGEMM_ERRORS_ENABLED
    if (simdcnn_debug_overlaps_f(A, size_A, C, size_C))
    {
        return SIMDCNN_SGEMM_MATRIX_ALIASING;
    }

    if (simdcnn_debug_overlaps_f(B, size_B, C, size_C))
    {
        return SIMDCNN_SGEMM_MATRIX_ALIASING;
    }

    if (((uint64_t)packed_A_bun % SIMDCNN_PAGE_ALIGNMENT != 0) ||
        ((uint64_t)packed_B_bun % SIMDCNN_PAGE_ALIGNMENT != 0))
    {
        return SIMDCNN_PACKING_ALIGNMENT_ERROR;
    }
#endif

    const size_t max_threads = omp_get_max_threads();
    const size_t num_threads = max_threads;

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

    const size_t packed_A_size = SIMDCNN_SGEMM_AVX2_PACKED_A_SIZE;
    const size_t packed_B_size = SIMDCNN_SGEMM_AVX2_PACKED_B_SIZE;

    for (size_t i = 0; i < num_threads; ++i)
    {
        if (!free_A)
        {
            packed_As[i] = &packed_A_bun[i * packed_A_size];
        }
        else
        {
            packed_As[i] = (float *)aligned_alloc(SIMDCNN_PAGE_ALIGNMENT, packed_A_size * sizeof(float));

            if (!packed_As[i])
            {
                simdcnn_sgemm_cleanup_avx2_(packed_As, packed_Bs, i, free_A, free_B);
                return SIMDCNN_SGEMM_OUT_OF_MEMORY;
            }
        }

        if (!free_B)
        {
            packed_Bs[i] = &packed_B_bun[i * packed_B_size];
        }
        else
        {
            packed_Bs[i] = (float *)aligned_alloc(SIMDCNN_PAGE_ALIGNMENT, packed_B_size * sizeof(float));
            if (!packed_Bs[i])
            {
                free(packed_As[i]);
                simdcnn_sgemm_cleanup_avx2_(packed_As, packed_Bs, i, free_A, free_B);
                return SIMDCNN_SGEMM_OUT_OF_MEMORY;
            }
        }
    }

    if (beta != 1.0f && beta != 0.0f)
    {
        for (size_t i = 0; i < M * N; ++i)
        {
            C[i] *= beta;
        }
    }
    else if (beta == 0.0f)
    {
        for (size_t i = 0; i < M * N; ++i)
        {
            C[i] = 0.0f;
        }
    }

#pragma omp parallel num_threads((int)num_threads)
    {
        size_t tid = omp_get_thread_num();
        float *packedA = packed_As[tid];
        float *packedB = packed_Bs[tid];

#pragma omp for schedule(dynamic)
        for (size_t ii = 0; ii < M; ii += SIMDCNN_SGEMM_AVX2_MC)
        {
            const size_t end_ii = SIMDCNN_MIN(ii + SIMDCNN_SGEMM_AVX2_MC, M);
            const size_t size_ii = end_ii - ii;

            for (size_t kk = 0; kk < K; kk += SIMDCNN_SGEMM_AVX2_KC)
            {
                const size_t end_kk = SIMDCNN_MIN(kk + SIMDCNN_SGEMM_AVX2_KC, K);
                const size_t size_kk = end_kk - kk;

                simdcnn_sgemm_pack_A_avx2_(packedA, A, K, ii, kk, size_ii, size_kk);

                for (size_t jj = 0; jj < N; jj += SIMDCNN_SGEMM_AVX2_NC)
                {
                    const size_t end_jj = SIMDCNN_MIN(jj + SIMDCNN_SGEMM_AVX2_NC, N);
                    const size_t size_jj = end_jj - jj;

                    simdcnn_sgemm_pack_B_avx2_(packedB, B, N, kk, jj, size_kk, size_jj, alpha);

                    for (size_t i = 0; i < (size_ii / SIMDCNN_SGEMM_AVX2_MR) * (SIMDCNN_SGEMM_AVX2_MR);
                         i += SIMDCNN_SGEMM_AVX2_MR)
                    {
                        for (size_t j = 0; j < (size_jj / SIMDCNN_SGEMM_AVX2_NR) * SIMDCNN_SGEMM_AVX2_NR;
                             j += SIMDCNN_SGEMM_AVX2_NR)
                        {
                            simdcnn_sgemm_ukernel_6x16_avx2(C + (ii + i) * N + (jj + j), N, packedA + i, packedB + j,
                                                            size_kk);
                        }
                    }

                    for (size_t i = 0; i < (size_ii / SIMDCNN_SGEMM_AVX2_MR) * SIMDCNN_SGEMM_AVX2_MR; ++i)
                    {
                        for (size_t j = (size_jj / SIMDCNN_SGEMM_AVX2_NR) * SIMDCNN_SGEMM_AVX2_NR; j < size_jj; ++j)
                        {
                            float sum = 0.0f;

                            for (size_t k = 0; k < size_kk; ++k)
                            {
                                sum += packedA[k * SIMDCNN_SGEMM_AVX2_MC + i] * packedB[k * SIMDCNN_SGEMM_AVX2_NC + j];
                            }

                            C[(ii + i) * N + (jj + j)] += sum;
                        }
                    }

                    for (size_t i = (size_ii / SIMDCNN_SGEMM_AVX2_MR) * SIMDCNN_SGEMM_AVX2_MR; i < size_ii; ++i)
                    {
                        for (size_t j = 0; j < size_jj; ++j)
                        {
                            float sum = 0.0f;

                            for (size_t k = 0; k < size_kk; ++k)
                            {
                                sum += packedA[k * SIMDCNN_SGEMM_AVX2_MC + i] * packedB[k * SIMDCNN_SGEMM_AVX2_NC + j];
                            }

                            C[(ii + i) * N + (jj + j)] += sum;
                        }
                    }
                }
            }
        }
    }

    simdcnn_sgemm_cleanup_avx2_(packed_As, packed_Bs, num_threads, free_A, free_B);

    return SIMDCNN_SGEMM_SUCCESS;
}