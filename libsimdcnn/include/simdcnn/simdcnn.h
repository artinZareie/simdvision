#pragma once

#include "simdcnn/errors.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define restrict
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    void printhello();

#ifdef HAVE_AVX2
    // Vector Addition
    /// vecadd: dst[i] = a[i] + b[i], 0 <= i < n
    extern void simdcnn_vecadd_f32_avx2(float *dst, float *a, float *b, uint64_t n);

    /// vecadd: dst[i] = a[i] + b[i], 0 <= i < n
    extern void simdcnn_vecadd_f64_avx2(double *dst, double *a, double *b, uint64_t n);

    // Activations
    /// relu: dst[i] = max(0.0f, a[i]), 0 <= i < n
    extern void simdcnn_relu_f32_avx2(float *dst, float *a, uint64_t n);

    /// relu: dst[i] = max(0.0f, a[i]), 0 <= i < n
    extern void simdcnn_relu_f64_avx2(double *dst, double *a, uint64_t n);

    /**
     * @brief Computes single‑precision general matrix multiplication (SGEMM)
     *        using a parallel, cache blocking algorithm with AVX2 intrinsics.
     *
     * @details
     * Performs the operation:
     *      C = alpha * A * B + beta * C
     * where A, B, and C are matrices stored in **row‑major** order.
     *
     * The algorithm uses a parallel approach combined with cache-blocking,
     * and vectorized operation.
     *
     * @note
     * **Aliasing Restriction**:
     * The matrices A and C must not overlap in memory, nor may B and C overlap.
     * Violating this condition leads to undefined behavior. The restriction exists
     * because packing reads from A and B while writing to C; overlapping regions
     * would cause data corruption.
     *
     * @param[out] C     Pointer to the output matrix (row‑major, dimensions M x N).
     * @param[in]  alpha Scalar multiplier for the product A * B.
     * @param[in]  beta  Scalar multiplier for the existing matrix C.
     * @param[in]  A     Pointer to the left input matrix (row‑major, dimensions M x K).
     * @param[in]  B     Pointer to the right input matrix (row‑major, dimensions K x N).
     * @param[in]  M     Number of rows of A and C.
     * @param[in]  K     Number of columns of A and rows of B.
     * @param[in]  N     Number of columns of B and C.
     *
     * @return simdcnn_sgemm_error_t
     *   - `SIMDCNN_SGEMM_SUCCESS`        Operation completed successfully.
     *   - `SIMDCNN_SGEMM_OUT_OF_MEMORY`  Failed to allocate internal packing buffers.
     *   - `SIMDCNN_SGEMM_MATRIX_ALIASING` Input and output matrices overlap
     *                                     (only when `SIMDCNN_SGEMM_ERRORS_ENABLED` is defined).
     *
     * @warning
     *   - The function assumes that the CPU supports AVX2 and FMA instructions.
     *   - All pointers must be suitably aligned for AVX2 loads/stores; misaligned
     *     accesses may cause segmentation faults or severe performance degradation.
     *
     * @see
     *   - `SIMDCNN_SGEMM_AVX2_AVX2_MC`, `SIMDCNN_SGEMM_AVX2_AVX2_KC`, `SIMDCNN_SGEMM_AVX2_AVX2_NC`
     *     for cache blocking parameters.
     *   - `simdcnn_debug_overlaps_f()` for the overlap detection routine.
     */

    extern simdcnn_sgemm_error_t simdcnn_sgemm_avx2(float *restrict C, float alpha, float beta, const float *A, const float *B,
                                             uint64_t M, uint64_t K, uint64_t N);
#endif

#ifdef __cplusplus
}
#endif