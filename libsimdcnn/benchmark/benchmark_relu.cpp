#include <benchmark/benchmark.h>
#include <ittnotify.h>
#include <random>
#include <simdcnn/simdcnn.h>
#include <vector>

#ifdef HAVE_ITTNOTIFY
#include <ittnotify.h>
#endif

#ifdef HAVE_AVX2

static void BM_Relu_F32_AVX2(benchmark::State &state)
{
#ifdef HAVE_ITTNOTIFY
    __itt_pause();
#endif
    const uint64_t n = state.range(0);
    std::vector<float> a(n);
    std::vector<float> dst(n, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (uint64_t i = 0; i < n; i++)
    {
        a[i] = dist(gen);
    }

#ifdef HAVE_ITTNOTIFY
    __itt_resume();
#endif
    for (auto _ : state)
    {
        simdcnn_relu_f32_avx2(dst.data(), a.data(), n);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

#ifdef HAVE_ITTNOTIFY
    __itt_pause();
#endif

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n) * sizeof(float) * 2);
}

BENCHMARK(BM_Relu_F32_AVX2)->Range(1 << 10, 1 << 24);

static void BM_Relu_F64_AVX2(benchmark::State &state)
{
#ifdef HAVE_ITTNOTIFY
    __itt_pause();
#endif
    const uint64_t n = state.range(0);
    std::vector<double> a(n);
    std::vector<double> dst(n, 0.0);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (uint64_t i = 0; i < n; i++)
    {
        a[i] = dist(gen);
    }

#ifdef HAVE_ITTNOTIFY
    __itt_resume();
#endif
    for (auto _ : state)
    {
        simdcnn_relu_f64_avx2(dst.data(), a.data(), n);
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }
#ifdef HAVE_ITTNOTIFY
    __itt_pause();
#endif

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n) * sizeof(double) * 2);
}

BENCHMARK(BM_Relu_F64_AVX2)->Range(1 << 10, 1 << 24);

#endif