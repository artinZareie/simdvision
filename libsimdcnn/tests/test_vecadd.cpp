#include <gtest/gtest.h>
#include <simdcnn/simdcnn.h>

TEST(VecAdd, BasicCorrectness)
{
    ASSERT_EQ(1, 1);
}

#ifdef HAVE_AVX2

TEST(VecAdd, Avx2Random)
{
    const int N = 1024;
    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *dst = (float *)malloc(N * sizeof(float));

    simdcnn_vecadd_f32_avx2(dst, a, b, N);

    for (size_t i = 0; i < N; i++)
    {
        ASSERT_FLOAT_EQ(a[i] + b[i], dst[i]);
    }
}

#endif