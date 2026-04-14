enable_language(CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "System GoogleTest not found; fetching via FetchContent.")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.0.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(simdcnn_test test_vecadd.cpp)

if(SIMD_CNN_AVX2_SUPPORTED)
    target_compile_definitions(simdcnn_test PRIVATE HAVE_AVX2=1)
endif()

if(SIMD_CNN_SSE2_SUPPORTED)
    target_compile_definitions(simdcnn_test PRIVATE HAVE_SSE2=1)
endif()

if(GTest_FOUND)
    target_link_libraries(simdcnn_test
        PRIVATE
        simdcnn
        GTest::gtest
        GTest::gtest_main
    )
else()
    target_link_libraries(simdcnn_test
        PRIVATE
        simdcnn
        gtest
        gtest_main
    )
endif()

include(GoogleTest)
gtest_discover_tests(simdcnn_test)