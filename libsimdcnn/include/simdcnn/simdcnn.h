#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void printhello();

#ifdef HAVE_AVX2
    extern void simdcnn_vecadd_f32_avx2(float *dst, float *a, float *b, size_t n);
#endif

#ifdef __cplusplus
}
#endif