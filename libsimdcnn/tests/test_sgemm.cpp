/// PAY ATTENTION: This version of this file is completely AI generated, and will be replaced by a handwritted version
/// in future.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <simdcnn/simdcnn.h>

#ifdef HAVE_AVX2

// ------------------------------------------------------------
// Reference implementation (naïve triple loop, row‑major)
// ------------------------------------------------------------
static void reference_sgemm(float *C, float alpha, float beta, const float *A, const float *B, uint64_t M, uint64_t K,
                            uint64_t N)
{
    // 1. Apply beta
    for (uint64_t i = 0; i < M * N; ++i)
        C[i] *= beta;

    // 2. Accumulate alpha * A * B
    for (uint64_t i = 0; i < M; ++i)
    {
        for (uint64_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (uint64_t k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] += alpha * sum;
        }
    }
}

// ------------------------------------------------------------
// Comparison helper with a reasonable tolerance
// ------------------------------------------------------------
static bool matrices_close(const float *C1, const float *C2, size_t len, float abs_tol = 2e-4f, float rel_tol = 2e-4f)
{
    for (size_t i = 0; i < len; ++i)
    {
        float diff = std::fabs(C1[i] - C2[i]);
        float scale = std::max(std::fabs(C1[i]), std::fabs(C2[i]));
        if (diff > abs_tol && diff > rel_tol * scale)
            return false;
    }
    return true;
}

// ------------------------------------------------------------
// Fill with deterministic random numbers
// ------------------------------------------------------------
static void fill_random(float *mat, size_t len, float lo = -1.0f, float hi = 1.0f)
{
    std::srand(12345);
    for (size_t i = 0; i < len; ++i)
    {
        float r = static_cast<float>(std::rand()) / RAND_MAX;
        mat[i] = lo + r * (hi - lo);
    }
}

// ------------------------------------------------------------
// Helper to run one test case
// ------------------------------------------------------------
static void run_test(uint64_t M, uint64_t K, uint64_t N, float alpha, float beta)
{
    size_t szA = M * K, szB = K * N, szC = M * N;

    float *A = (float *)aligned_alloc(64, szA * sizeof(float));
    float *B = (float *)aligned_alloc(64, szB * sizeof(float));
    float *C1 = (float *)aligned_alloc(64, szC * sizeof(float));
    float *C2 = (float *)aligned_alloc(64, szC * sizeof(float));

    ASSERT_NE(A, nullptr);
    ASSERT_NE(B, nullptr);
    ASSERT_NE(C1, nullptr);
    ASSERT_NE(C2, nullptr);

    fill_random(A, szA);
    fill_random(B, szB);
    fill_random(C1, szC);
    std::copy(C1, C1 + szC, C2);

    // Run your kernel
    simdcnn_sgemm_error_t err = simdcnn_sgemm_avx2(C1, alpha, beta, A, B, M, K, N);
    ASSERT_EQ(err, SIMDCNN_SGEMM_SUCCESS);

    // Run reference
    reference_sgemm(C2, alpha, beta, A, B, M, K, N);

    EXPECT_TRUE(matrices_close(C1, C2, szC))
        << "Mismatch: M=" << M << " K=" << K << " N=" << N << " alpha=" << alpha << " beta=" << beta;

    free(A);
    free(B);
    free(C1);
    free(C2);
}

// =========================================================================
//  The actual tests – only a handful, covering critical cases
// =========================================================================

// 1. Tiny sizes, non‑multiples of any block, beta = 0
TEST(Sgemm, Small_3x5x7_beta0)
{
    run_test(3, 5, 7, 1.0f, 0.0f);
}

// 2. Same but with beta = 1 (C is already scaled)
TEST(Sgemm, Small_4x6x9_beta1)
{
    run_test(4, 6, 9, 1.0f, 1.0f);
}

// 3. Exact micro‑tile sizes (6x16) – exercises the fast path
TEST(Sgemm, MicroTile_6x16x16)
{
    run_test(6, 16, 16, 1.0f, 0.0f);
}

// 4. Slightly larger than micro‑tile, non‑multiples
TEST(Sgemm, NonMultiple_7x17x13)
{
    run_test(7, 17, 13, 2.5f, 1.0f);
}

// 5. Exercise alpha = 0 (should only apply beta)
TEST(Sgemm, AlphaZero_beta2)
{
    run_test(8, 8, 8, 0.0f, 2.0f);
}

// 6. Reasonably large square matrix – touches threading and cache blocking
TEST(Sgemm, Medium_256x256x256)
{
    run_test(256, 256, 256, 1.0f, 0.0f);
}

// 7. Large non‑multiple of block sizes – stresses cleanup code
TEST(Sgemm, BlockBoundary_385x255x4095)
{
    run_test(385, 255, 4095, 0.5f, 1.0f);
}

// 8. Exact macro‑tile size (if you defined MC=384, KC=256, NC=4096)
TEST(Sgemm, ExactBlockSizes)
{
    run_test(384, 256, 4096, 1.0f, 0.0f);
}

// 9. Very small K (edge)
TEST(Sgemm, K_eq_1)
{
    run_test(15, 1, 23, 2.0f, 0.0f);
}

// 10. Very small M and N, but large K
TEST(Sgemm, TallK_2x100x2)
{
    run_test(2, 100, 2, 1.0f, 1.0f);
}

#endif // HAVE_AVX2