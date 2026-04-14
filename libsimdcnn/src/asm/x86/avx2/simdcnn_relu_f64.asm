[BITS 64]
%define public_prefix simdcnn
%define private_prefix simdcnn
%include "x86inc.asm"

INIT_YMM avx2

section .text
; simdcnn_relu_f64_avx2(dst, a, b, n)
; dst: pointer
; a  : pointer
; n  : size_t
; dst = relu(a)
ALIGN 16
cvisible relu_f64, 3, 5, 9, dst, a, n, end_ptr, orig_dst
    test nq, nq ; If n == 0, exit
    jz .end

    vxorps ymm8, ymm8, ymm8 ; ymm8 = 0, used for vmaxps

    mov end_ptrq, nq
    and end_ptrq, -16  ; Each iteration covers 8 * 4 = 32 elements
    shl end_ptrq, 3
    add end_ptrq, dstq ; dst's end pointer (n_vec)
    
    mov orig_dstq, dstq

ALIGN 16
.loop_vectorized:
    cmp dstq, end_ptrq
    jae .loop_vectorized_done

    vmovupd ymm0, [aq]
    vmovupd ymm1, [aq + 32]
    vmovupd ymm2, [aq + 64]
    vmovupd ymm3, [aq + 96]

    vmaxpd ymm0, ymm0, ymm8
    vmaxpd ymm1, ymm1, ymm8
    vmaxpd ymm2, ymm2, ymm8
    vmaxpd ymm3, ymm3, ymm8

    vmovupd [dstq], ymm0
    vmovupd [dstq + 32], ymm1
    vmovupd [dstq + 64], ymm2
    vmovupd [dstq + 96], ymm3

    add aq, 32 * 4
    add dstq, 32 * 4
    jmp .loop_vectorized

.loop_vectorized_done:

    mov end_ptrq, nq
    shl end_ptrq, 3
    add end_ptrq, orig_dstq

ALIGN 16
.loop_vectorized_tail:
    cmp dstq, end_ptrq
    jae .loop_vectorized_tail_done

    vmovsd xmm0, [aq]
    vmaxpd xmm0, xmm0, xmm8
    vmovsd [dstq], xmm0

    add aq, 8
    add dstq, 8
    jmp .loop_vectorized_tail

.loop_vectorized_tail_done:

.end:

    vzeroupper
    RET