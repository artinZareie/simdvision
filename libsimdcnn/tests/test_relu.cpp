#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <simdcnn/simdcnn.h>

#ifdef HAVE_AVX2

TEST(Relu, Avx2F32Random)
{
    srand(time(NULL));

    const int N = 1 << 16;
    float *a = (float *)malloc(N * sizeof(float));
    float *dst = (float *)malloc(N * sizeof(float));

    for (size_t i = 0; i < N; i++)
    {
        a[i] = (rand() / (float)RAND_MAX) - 0.5f;
        dst[i] = 0.0f;
    }

    simdcnn_relu_f32_avx2(dst, a, N);

    for (size_t i = 0; i < N; i++)
    {
        ASSERT_FLOAT_EQ(std::max(0.0f, a[i]), dst[i]);
    }

    free(a);
    free(dst);
}

TEST(Relu, Avx2F64Random)
{
    srand(time(NULL));

    const int N = 1 << 16;
    double *a = (double *)malloc(N * sizeof(double));
    double *dst = (double *)malloc(N * sizeof(double));

    for (size_t i = 0; i < N; i++)
    {
        a[i] = (rand() / (double)RAND_MAX) - 0.5f;
        dst[i] = 0.0f;
    }

    simdcnn_relu_f64_avx2(dst, a, N);

    for (size_t i = 0; i < N; i++)
    {
        ASSERT_DOUBLE_EQ(std::max(0.0, a[i]), dst[i]);
    }

    free(a);
    free(dst);
}

#endif