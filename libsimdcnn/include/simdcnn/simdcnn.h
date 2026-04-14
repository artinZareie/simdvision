#pragma once

#include <stddef.h>
#include <stdint.h>

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
#endif

#ifdef __cplusplus
}
#endif