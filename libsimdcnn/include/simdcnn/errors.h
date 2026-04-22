#pragma once

typedef enum
{
    SIMDCNN_SGEMM_SUCCESS = 0,
    SIMDCNN_SGEMM_OUT_OF_MEMORY,
    SIMDCNN_SGEMM_MATRIX_ALIASING
} simdcnn_sgemm_error_t;