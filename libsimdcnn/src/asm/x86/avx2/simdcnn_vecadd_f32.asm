[BITS 64]
%define public_prefix simdcnn
%define private_prefix simdcnn
%include "x86inc.asm"

INIT_YMM avx2

section .text
; simdcnn_vecadd_f32_avx2(dst, a, b, n)
; dst: pointer (float[n])
; a  : pointer (float[n])
; b  : pointer (float[n])
; n  : uint64_t
; dst = a + b
ALIGN 16
cvisible vecadd_f32, 4, 6, 8, dst, a, b, n, end_ptr, orig_dst
    test nq, nq ; If n == 0, exit
    jz .end

    ; Alignment prologue: Aligns dsq to 32-byte address
    ; by reading until a 32-byte address.
    mov end_ptrq, dstq
    add end_ptrq, 31
    and end_ptrq, -32

.align_loop:
    cmp dstq, end_ptrq
    jae .align_loop_done

    test nq, nq
    jz .end

    vmovss xmm0, [aq]
    vmovss xmm1, [bq]
    vaddss xmm0, xmm0, xmm1
    vmovss [dstq], xmm0

    add aq, 4
    add bq, 4
    add dstq, 4
    sub nq, 1

    jmp .align_loop

.align_loop_done:

    mov end_ptrq, nq
    and end_ptrq, -32 ; Each iteration covers 8 * 4 = 32 elements
    shl end_ptrq, 2
    add end_ptrq, dstq ; dst's end pointer (n_vec)
    
    mov orig_dstq, dstq

ALIGN 16
.loop_vectorized:
    cmp dstq, end_ptrq
    jae .loop_vectorized_done

    vmovups ymm0, [aq]
    vmovups ymm1, [bq]

    vmovups ymm2, [aq + 32]
    vmovups ymm3, [bq + 32]

    vmovups ymm4, [aq + 64]
    vmovups ymm5, [bq + 64]

    vmovups ymm6, [aq + 96]
    vmovups ymm7, [bq + 96]

    vaddps ymm0, ymm0, ymm1
    vaddps ymm2, ymm2, ymm3
    vaddps ymm4, ymm4, ymm5
    vaddps ymm6, ymm6, ymm7

    vmovntps [dstq], ymm0
    vmovntps [dstq + 32], ymm2
    vmovntps [dstq + 64], ymm4
    vmovntps [dstq + 96], ymm6

    add aq, 32 * 4
    add bq, 32 * 4
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
    vmovss xmm1, [bq]
    vaddss xmm0, xmm0, xmm1
    vmovss [dstq], xmm0

    add aq, 4
    add bq, 4
    add dstq, 4
    jmp .loop_vectorized_tail

.loop_vectorized_tail_done:

.end:

    vzeroupper
    RET