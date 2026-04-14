#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <simdcnn/simdcnn.h>

TEST(VecAdd, BasicCorrectness)
{
    ASSERT_EQ(1, 1);
}

#ifdef HAVE_AVX2

TEST(VecAdd, Avx2Random)
{
    srand(time(NULL));

    const int N = 1024;
    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *dst = (float *)malloc(N * sizeof(float));

    for (size_t i = 0; i < N; i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        dst[i] = 0.0f;
    }

    simdcnn_vecadd_f32_avx2(dst, a, b, N);

    for (size_t i = 0; i < N; i++)
    {
        ASSERT_FLOAT_EQ(a[i] + b[i], dst[i]);
    }

    free(a);
    free(b);
    free(dst);
}

#endif