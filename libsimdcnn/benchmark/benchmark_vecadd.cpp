#include <benchmark/benchmark.h>
#include <simdcnn/simdcnn.h>
#include <vector>

#ifdef HAVE_ITTNOTIFY
#include <ittnotify.h>
#endif

#ifdef HAVE_AVX2

static void BM_VecAdd_F32_AVX2(benchmark::State &state)
{
#ifdef HAVE_ITTNOTIFY
    __itt_pause();
#endif
    const uint64_t n = state.range(0);
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> dst(n, 0.0f);

#ifdef HAVE_ITTNOTIFY
    __itt_resume();
#endif
    for (auto _ : state)
    {
        simdcnn_vecadd_f32_avx2(dst.data(), a.data(), b.data(), n);
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }
#ifdef HAVE_ITTNOTIFY
    __itt_pause();
#endif

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n) * sizeof(float) * 3);
}

BENCHMARK(BM_VecAdd_F32_AVX2)->Range(1 << 10, 1 << 24);

static void BM_VecAdd_F64_AVX2(benchmark::State &state)
{
#ifdef HAVE_ITTNOTIFY
    __itt_pause();
#endif
    const uint64_t n = state.range(0);
    std::vector<double> a(n, 1.0);
    std::vector<double> b(n, 2.0);
    std::vector<double> dst(n, 0.0);

#ifdef HAVE_ITTNOTIFY
    __itt_resume();
#endif
    for (auto _ : state)
    {
        simdcnn_vecadd_f64_avx2(dst.data(), a.data(), b.data(), n);
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }
#ifdef HAVE_ITTNOTIFY
    __itt_pause();
#endif

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n) * sizeof(double) * 3);
}

BENCHMARK(BM_VecAdd_F64_AVX2)->Range(1 << 10, 1 << 24);

#endif