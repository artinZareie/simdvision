enable_language(CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    message(STATUS "System Google Benchmark not found; fetching via FetchContent.")
    include(FetchContent)

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable testing of the benchmark library" FORCE)

    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
    )
    FetchContent_MakeAvailable(googlebenchmark)
endif()

set(SIMDCNN_BENCHMARK_SOURCES
    main.cpp

    benchmark_vecadd.cpp
    benchmark_relu.cpp
)

add_executable(simdcnn_benchmark
    ${SIMDCNN_BENCHMARK_SOURCES}
)

if(SIMD_CNN_AVX2_SUPPORTED)
    target_compile_definitions(simdcnn_benchmark PRIVATE HAVE_AVX2=1)
endif()

if(SIMD_CNN_SSE2_SUPPORTED)
    target_compile_definitions(simdcnn_benchmark PRIVATE HAVE_SSE2=1)
endif()

find_package(Threads REQUIRED)

if(benchmark_FOUND)
    target_link_libraries(simdcnn_benchmark
        PRIVATE
        simdcnn
        benchmark::benchmark
        benchmark::benchmark_main
        Threads::Threads
    )
else()
    target_link_libraries(simdcnn_benchmark
        PRIVATE
        simdcnn
        benchmark
        benchmark_main
        Threads::Threads
    )
endif()

if(SIMD_CNN_VTUNE_SUPPORT AND SIMD_CNN_VTUNE_AVAILABLE)
    target_include_directories(simdcnn_benchmark PRIVATE ${ITTNOTIFY_INCLUDE_DIR})
    target_link_libraries(simdcnn_benchmark PRIVATE ${ITTNOTIFY_LIBRARY})

    target_compile_definitions(simdcnn_benchmark PRIVATE HAVE_ITTNOTIFY=1)
endif()
