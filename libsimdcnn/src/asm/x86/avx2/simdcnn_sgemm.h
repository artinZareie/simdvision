#pragma once

#include <stddef.h>
#include <stdint.h>

// TODO: To be replaced with efficient values
#define SIMDCNN_MATMUL_AVX2_MC 128 /// Macro-tile size in M dimension
#define SIMDCNN_MATMUL_AVX2_NC 128 /// Macro-tile size in N dimension
#define SIMDCNN_MATMUL_AVX2_KC 256 /// Macro-tile size in K dimension

#define SIMDCNN_MATMUL_AVX2_MR 8  /// Micro-tile size in M dimension
#define SIMDCNN_MATMUL_AVX2_NR 12 /// Micro-tile size in N dimension

#define SIMDCNN_PAGE_ALIGNMENT 4096
#define SIMDCNN_MATMUL_AVX2_MAX_THREADS 128