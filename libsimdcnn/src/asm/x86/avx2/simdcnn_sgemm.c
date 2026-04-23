#include "simdcnn_sgemm.h"
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <simdcnn/debug.h>
#include <simdcnn/def.h>
#include <simdcnn/errors.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/// Cleans up memory allocated for simdcnn_sgemm_avx2
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
                                       size_t size_jj)
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
                packedB[j + i * SIMDCNN_SGEMM_AVX2_NC] = B[(i + kk) * N + j + jj];
            }
        }
    }
}

/// See header for fulll documentation.
simdcnn_sgemm_error_t simdcnn_sgemm_avx2(float *restrict C, float alpha, float beta, const float *A, const float *B,
                                         uint64_t M, uint64_t K, uint64_t N)
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

    const size_t packed_A_size =
        (SIMDCNN_SGEMM_AVX2_MC * SIMDCNN_SGEMM_AVX2_KC + (SIMDCNN_PAGE_ALIGNMENT - 1)) & ~(SIMDCNN_PAGE_ALIGNMENT - 1);

    const size_t packed_B_size =
        (SIMDCNN_SGEMM_AVX2_NC * SIMDCNN_SGEMM_AVX2_KC + (SIMDCNN_PAGE_ALIGNMENT - 1)) & ~(SIMDCNN_PAGE_ALIGNMENT - 1);

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

        if (beta != 1.0f && beta != 0.0f)
        {
#pragma omp for schedule(static)
            for (size_t i = 0; i < M * N; ++i)
            {
                C[i] *= beta;
            }
        }
        else if (beta == 0.0f)
        {
#pragma omp for schedule(static)
            for (size_t i = 0; i < M * N; ++i)
            {
                C[i] = 0.0f;
            }
        }

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

                    simdcnn_sgemm_pack_B_avx2_(packedB, B, N, kk, jj, size_kk, size_jj);

                    for (size_t i = 0; i < (size_ii / SIMDCNN_SGEMM_AVX2_MR) * (SIMDCNN_SGEMM_AVX2_MR);
                         i += SIMDCNN_SGEMM_AVX2_MR)
                    {
                        for (size_t j = 0; j < (size_jj / SIMDCNN_SGEMM_AVX2_NR) * SIMDCNN_SGEMM_AVX2_NR;
                             j += SIMDCNN_SGEMM_AVX2_NR)
                        {
                            __m256 c00;
                            __m256 c01;
                            __m256 c10;
                            __m256 c11;
                            __m256 c20;
                            __m256 c21;
                            __m256 c30;
                            __m256 c31;
                            __m256 c40;
                            __m256 c41;
                            __m256 c50;
                            __m256 c51;

                            c00 = _mm256_loadu_ps(C + (i + ii + 0) * N + j + jj);
                            c01 = _mm256_loadu_ps(C + (i + ii + 0) * N + j + jj + 8);
                            c10 = _mm256_loadu_ps(C + (i + ii + 1) * N + j + jj);
                            c11 = _mm256_loadu_ps(C + (i + ii + 1) * N + j + jj + 8);
                            c20 = _mm256_loadu_ps(C + (i + ii + 2) * N + j + jj);
                            c21 = _mm256_loadu_ps(C + (i + ii + 2) * N + j + jj + 8);
                            c30 = _mm256_loadu_ps(C + (i + ii + 3) * N + j + jj);
                            c31 = _mm256_loadu_ps(C + (i + ii + 3) * N + j + jj + 8);
                            c40 = _mm256_loadu_ps(C + (i + ii + 4) * N + j + jj);
                            c41 = _mm256_loadu_ps(C + (i + ii + 4) * N + j + jj + 8);
                            c50 = _mm256_loadu_ps(C + (i + ii + 5) * N + j + jj);
                            c51 = _mm256_loadu_ps(C + (i + ii + 5) * N + j + jj + 8);

                            const float *a_packed = packedA + i;
                            const float *b_packed = packedB + j;

#pragma GCC unroll 4
                            for (size_t k = 0; k < size_kk; ++k)
                            {

                                __m256 b0 = _mm256_load_ps(b_packed);
                                __m256 b1 = _mm256_load_ps(b_packed + 8);

                                __m256 a0 = _mm256_broadcast_ss(a_packed);
                                c00 = _mm256_fmadd_ps(a0, b0, c00);
                                c01 = _mm256_fmadd_ps(a0, b1, c01);

                                __m256 a1 = _mm256_broadcast_ss(a_packed + 1);
                                c10 = _mm256_fmadd_ps(a1, b0, c10);
                                c11 = _mm256_fmadd_ps(a1, b1, c11);

                                __m256 a2 = _mm256_broadcast_ss(a_packed + 2);
                                c20 = _mm256_fmadd_ps(a2, b0, c20);
                                c21 = _mm256_fmadd_ps(a2, b1, c21);

                                __m256 a3 = _mm256_broadcast_ss(a_packed + 3);
                                c30 = _mm256_fmadd_ps(a3, b0, c30);
                                c31 = _mm256_fmadd_ps(a3, b1, c31);

                                __m256 a4 = _mm256_broadcast_ss(a_packed + 4);
                                c40 = _mm256_fmadd_ps(a4, b0, c40);
                                c41 = _mm256_fmadd_ps(a4, b1, c41);

                                __m256 a5 = _mm256_broadcast_ss(a_packed + 5);
                                c50 = _mm256_fmadd_ps(a5, b0, c50);
                                c51 = _mm256_fmadd_ps(a5, b1, c51);

                                a_packed += SIMDCNN_SGEMM_AVX2_MC;
                                b_packed += SIMDCNN_SGEMM_AVX2_NC;
                            }

                            __m256 valpha = _mm256_set1_ps(alpha);
                            c00 = _mm256_mul_ps(c00, valpha);
                            c01 = _mm256_mul_ps(c01, valpha);
                            c10 = _mm256_mul_ps(c10, valpha);
                            c11 = _mm256_mul_ps(c11, valpha);
                            c20 = _mm256_mul_ps(c20, valpha);
                            c21 = _mm256_mul_ps(c21, valpha);
                            c30 = _mm256_mul_ps(c30, valpha);
                            c31 = _mm256_mul_ps(c31, valpha);
                            c40 = _mm256_mul_ps(c40, valpha);
                            c41 = _mm256_mul_ps(c41, valpha);
                            c50 = _mm256_mul_ps(c50, valpha);
                            c51 = _mm256_mul_ps(c51, valpha);

                            _mm256_storeu_ps(C + (i + ii + 0) * N + j + jj, c00);
                            _mm256_storeu_ps(C + (i + ii + 0) * N + j + jj + 8, c01);
                            _mm256_storeu_ps(C + (i + ii + 1) * N + j + jj, c10);
                            _mm256_storeu_ps(C + (i + ii + 1) * N + j + jj + 8, c11);
                            _mm256_storeu_ps(C + (i + ii + 2) * N + j + jj, c20);
                            _mm256_storeu_ps(C + (i + ii + 2) * N + j + jj + 8, c21);
                            _mm256_storeu_ps(C + (i + ii + 3) * N + j + jj, c30);
                            _mm256_storeu_ps(C + (i + ii + 3) * N + j + jj + 8, c31);
                            _mm256_storeu_ps(C + (i + ii + 4) * N + j + jj, c40);
                            _mm256_storeu_ps(C + (i + ii + 4) * N + j + jj + 8, c41);
                            _mm256_storeu_ps(C + (i + ii + 5) * N + j + jj, c50);
                            _mm256_storeu_ps(C + (i + ii + 5) * N + j + jj + 8, c51);
                        }

                        for (size_t j = (size_jj / SIMDCNN_SGEMM_AVX2_NR) * SIMDCNN_SGEMM_AVX2_NR; j < size_jj; ++j)
                        {
                            float sum = 0.0f;

                            for (size_t k = 0; k < size_kk; ++k)
                            {
                                sum += packedA[k * SIMDCNN_SGEMM_AVX2_MC + i] * packedB[k * SIMDCNN_SGEMM_AVX2_NC + j];
                            }

                            C[(ii + i) * N + (jj + j)] += alpha * sum;
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

                            C[(ii + i) * N + (jj + j)] += alpha * sum;
                        }
                    }
                }
            }
        }
    }

    simdcnn_sgemm_cleanup_avx2_(packed_As, packed_Bs, num_threads);

    return SIMDCNN_SGEMM_SUCCESS;
}