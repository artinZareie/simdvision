[BITS 64]
%define public_prefix simdcnn
%define private_prefix simdcnn
%include "x86inc.asm"


%define SIMDCNN_SGEMM_AVX2_MC 384 
%define SIMDCNN_SGEMM_AVX2_NC 4096
%define SIMDCNN_SGEMM_AVX2_KC 256

INIT_YMM avx2

section .text
; void simdcnn_sgemm_ukernel_6x16_avx2(float *c, uint64_t N, const float *a, const float *b, uint64_t size_kk);
ALIGN 16
cglobal sgemm_ukernel_6x16, 5, 6, 15, c, N, a, b, size_kk

    test size_kkq, size_kkq
    jz .end

    %define c00 ymm0
    %define c01 ymm1
    %define c10 ymm2
    %define c11 ymm3
    %define c20 ymm4
    %define c21 ymm5
    %define c30 ymm6
    %define c31 ymm7
    %define c40 ymm8
    %define c41 ymm9
    %define c50 ymm10
    %define c51 ymm11

    %define b0 ymm12
    %define b1 ymm13

    %define a_tmp ymm14

    vmovups c00, [cq]
    vmovups c01, [cq + 32]
    lea r5q, [cq + Nq * 4]

    vmovups c10, [r5q]
    vmovups c11, [r5q + 32]
    lea r5q, [r5q + Nq * 4]

    vmovups c20, [r5q]
    vmovups c21, [r5q + 32]
    lea r5q, [r5q + Nq * 4]

    vmovups c30, [r5q]
    vmovups c31, [r5q + 32]
    lea r5q, [r5q + Nq * 4]

    vmovups c40, [r5q]
    vmovups c41, [r5q + 32]
    lea r5q, [r5q + Nq * 4]

    vmovups c50, [r5q]
    vmovups c51, [r5q + 32]

.loop_k:
    vmovaps b0, [bq]
    vmovaps b1, [bq + 32]

    vbroadcastss a_tmp, [aq]
    vfmadd231ps c00, a_tmp, b0
    vfmadd231ps c01, a_tmp, b1

    vbroadcastss a_tmp, [aq + 4]
    vfmadd231ps c10, a_tmp, b0
    vfmadd231ps c11, a_tmp, b1

    vbroadcastss a_tmp, [aq + 8]
    vfmadd231ps c20, a_tmp, b0
    vfmadd231ps c21, a_tmp, b1

    vbroadcastss a_tmp, [aq + 12]
    vfmadd231ps c30, a_tmp, b0
    vfmadd231ps c31, a_tmp, b1

    vbroadcastss a_tmp, [aq + 16]
    vfmadd231ps c40, a_tmp, b0
    vfmadd231ps c41, a_tmp, b1

    vbroadcastss a_tmp, [aq + 20]
    vfmadd231ps c50, a_tmp, b0
    vfmadd231ps c51, a_tmp, b1

    add aq, SIMDCNN_SGEMM_AVX2_MC * 4
    add bq, SIMDCNN_SGEMM_AVX2_NC * 4

    dec size_kkq
    jnz .loop_k

    vmovups [cq], c00
    vmovups [cq + 32], c01
    lea r5q, [cq + Nq * 4]
    vmovups [r5q], c10
    vmovups [r5q + 32], c11
    lea r5q, [r5q + Nq * 4]
    vmovups [r5q], c20
    vmovups [r5q + 32], c21
    lea r5q, [r5q + Nq * 4]
    vmovups [r5q], c30
    vmovups [r5q + 32], c31
    lea r5q, [r5q + Nq * 4]
    vmovups [r5q], c40
    vmovups [r5q + 32], c41
    lea r5q, [r5q + Nq * 4]
    vmovups [r5q], c50
    vmovups [r5q + 32], c51

.end:
    vzeroupper
    RET