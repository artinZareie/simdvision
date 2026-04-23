[BITS 64]
%define public_prefix simdcnn
%define private_prefix simdcnn
%include "x86inc.asm"

INIT_YMM avx2

section .text
; simdcnn_vecadd_f64_avx2(dst, a, b, n)
; dst: pointer (double[n])
; a  : pointer (double[n])
; b  : pointer (double[n])
; n  : uint64_t
; dst = a + b
ALIGN 32
cvisible vecadd_f64, 4, 6, 8, dst, a, b, n, end_ptr, orig_dst
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

    vmovsd xmm0, [aq]
    vmovsd xmm1, [bq]
    vaddsd xmm0, xmm0, xmm1
    vmovsd [dstq], xmm0

    add aq, 8
    add bq, 8
    add dstq, 8
    sub nq, 1

    jmp .align_loop

.align_loop_done:

    mov end_ptrq, nq
    and end_ptrq, -16  ; Each iteration covers 4 * 4 = 16 elements
    shl end_ptrq, 3
    add end_ptrq, dstq ; dst's end pointer (n_vec)
    
    mov orig_dstq, dstq

ALIGN 32
.loop_vectorized:
    cmp dstq, end_ptrq
    jae .loop_vectorized_done

    vmovupd ymm0, [aq]
    vmovupd ymm1, [bq]

    vmovupd ymm2, [aq + 32]
    vmovupd ymm3, [bq + 32]

    vmovupd ymm4, [aq + 64]
    vmovupd ymm5, [bq + 64]

    vmovupd ymm6, [aq + 96]
    vmovupd ymm7, [bq + 96]

    vaddpd ymm0, ymm0, ymm1
    vaddpd ymm2, ymm2, ymm3
    vaddpd ymm4, ymm4, ymm5
    vaddpd ymm6, ymm6, ymm7

    vmovapd [dstq], ymm0
    vmovapd [dstq + 32], ymm2
    vmovapd [dstq + 64], ymm4
    vmovapd [dstq + 96], ymm6

    add aq, 32 * 4
    add bq, 32 * 4
    add dstq, 32 * 4
    jmp .loop_vectorized

.loop_vectorized_done:

    mov end_ptrq, nq
    shl end_ptrq, 3
    add end_ptrq, orig_dstq

ALIGN 32
.loop_vectorized_tail:
    cmp dstq, end_ptrq
    jae .loop_vectorized_tail_done

    vmovsd xmm0, [aq]
    vmovsd xmm1, [bq]
    vaddsd xmm0, xmm0, xmm1
    vmovsd [dstq], xmm0

    add aq, 8
    add bq, 8
    add dstq, 8
    jmp .loop_vectorized_tail

.loop_vectorized_tail_done:

.end:

    vzeroupper
    RET