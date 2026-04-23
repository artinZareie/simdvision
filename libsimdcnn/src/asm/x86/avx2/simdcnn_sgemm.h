#pragma once

#include <stddef.h>
#include <stdint.h>

// TODO: To be replaced with efficient values
#define SIMDCNN_SGEMM_AVX2_MC 384  /// Macro-tile size in M dimension
#define SIMDCNN_SGEMM_AVX2_NC 4096 /// Macro-tile size in N dimension
#define SIMDCNN_SGEMM_AVX2_KC 256  /// Macro-tile size in K dimension

#define SIMDCNN_SGEMM_AVX2_MR 6  /// Micro-tile size in M dimension
#define SIMDCNN_SGEMM_AVX2_NR 16 /// Micro-tile size in N dimension

#define SIMDCNN_PAGE_ALIGNMENT 4096