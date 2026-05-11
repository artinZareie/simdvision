#pragma once

#include <stddef.h>
#include <stdint.h>

#define SIMDCNN_PAGE_ALIGNMENT 4096

#ifdef HAVE_AVX2
// TODO: To be replaced with efficient values
#define SIMDCNN_SGEMM_AVX2_MC 384  /// Macro-tile size in M dimension
#define SIMDCNN_SGEMM_AVX2_NC 4096 /// Macro-tile size in N dimension
#define SIMDCNN_SGEMM_AVX2_KC 256  /// Macro-tile size in K dimension

#define SIMDCNN_SGEMM_AVX2_MR 6  /// Micro-tile size in M dimension
#define SIMDCNN_SGEMM_AVX2_NR 16 /// Micro-tile size in N dimension

#define SIMDCNN_SGEMM_AVX2_PACKED_A_SIZE                                                                                \
    ((SIMDCNN_SGEMM_AVX2_MC * SIMDCNN_SGEMM_AVX2_KC + (SIMDCNN_PAGE_ALIGNMENT - 1)) & ~(SIMDCNN_PAGE_ALIGNMENT - 1))

#define SIMDCNN_SGEMM_AVX2_PACKED_B_SIZE                                                                               \
    (SIMDCNN_SGEMM_AVX2_NC * SIMDCNN_SGEMM_AVX2_KC + (SIMDCNN_PAGE_ALIGNMENT - 1)) & ~(SIMDCNN_PAGE_ALIGNMENT - 1)
#endif