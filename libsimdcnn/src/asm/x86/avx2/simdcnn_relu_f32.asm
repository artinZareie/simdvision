[BITS 64]
%define public_prefix simdcnn
%define private_prefix simdcnn
%include "x86inc.asm"

INIT_YMM avx2

section .text
; simdcnn_relu_f32_avx2(dst, a, b, n)
; dst: pointer float[n]
; a  : pointer float[n]
; n  : size_t
; dst = relu(a)
ALIGN 16
cvisible relu_f32, 3, 5, 9, dst, a, n, end_ptr, orig_dst
    test nq, nq ; If n == 0, exit
    jz .end

    vxorps ymm8, ymm8, ymm8 ; ymm8 = 0, used for vmaxps

    ; Aignment Prologue: Process elements one by one until dq is 32-byte aligned
    mov end_ptrq, dstq
    add end_ptrq, 31
    and end_ptrq, -32

.align_loop:
    cmp dstq, end_ptrq
    jae .align_loop_done

    test nq, nq
    jz .end

    vmovss xmm0, [aq]
    vmaxss xmm0, xmm0, xmm8
    vmovss [dstq], xmm0

    add aq, 4
    add dstq, 4
    sub nq, 1

    jmp .align_loop

.align_loop_done:

    mov end_ptrq, nq
    and end_ptrq, -32  ; Each iteration covers 8 * 4 = 32 elements
    shl end_ptrq, 2
    add end_ptrq, dstq ; dst's end pointer (n_vec)
    
    mov orig_dstq, dstq

ALIGN 16
.loop_vectorized:
    cmp dstq, end_ptrq
    jae .loop_vectorized_done

    vmovups ymm0, [aq]
    vmovups ymm1, [aq + 32]
    vmovups ymm2, [aq + 64]
    vmovups ymm3, [aq + 96]

    vmaxps ymm0, ymm0, ymm8
    vmaxps ymm1, ymm1, ymm8
    vmaxps ymm2, ymm2, ymm8
    vmaxps ymm3, ymm3, ymm8

    vmovntps [dstq], ymm0
    vmovntps [dstq + 32], ymm1
    vmovntps [dstq + 64], ymm2
    vmovntps [dstq + 96], ymm3

    add aq, 32 * 4
    add dstq, 32 * 4
    jmp .loop_vectorized

.loop_vectorized_done:

    mov end_ptrq, nq
    shl end_ptrq, 2
    add end_ptrq, orig_dstq

ALIGN 16
.loop_vectorized_tail:
    cmp dstq, end_ptrq
    jae .loop_vectorized_tail_done

    vmovss xmm0, [aq]
    vmaxps xmm0, xmm0, xmm8
    vmovss [dstq], xmm0

    add aq, 4
    add dstq, 4
    jmp .loop_vectorized_tail

.loop_vectorized_tail_done:

.end:

    vzeroupper
    RET