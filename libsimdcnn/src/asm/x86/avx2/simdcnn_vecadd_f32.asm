[BITS 64]
%define private_prefix x86_vecadd
%include "x86inc.asm"

GLOBAL simdcnn_vecadd_f32_avx2

section .text
; simdcnn_vecadd_f32_avx2(dst, a, b, n)
; dst: pointer
; a  : pointer
; b  : pointer
; n  : size_t
ALIGN 16
simdcnn_vecadd_f32_avx2:
cglobal simdcnn_vecadd_f32_avx2, 4, 6, 0, dst, a, b, n
    sub rsp, 64
    vmovups [rsp + 0], ymm6
    vmovups [rsp + 32], ymm7

    test nq, nq ; If n == 0, exit
    jz .end

    mov r4, nq
    and r4, -32
    shl r4, 2
    add r4, dstq ; dst's end pointer (n_vec)
    
    mov r5, dstq

ALIGN 16
.loop_vectorized:
    cmp dstq, r4
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

    vmovups [dstq], ymm0
    vmovups [dstq + 32], ymm2
    vmovups [dstq + 64], ymm4
    vmovups [dstq + 96], ymm6

    add aq, 32 * 4
    add bq, 32 * 4
    add dstq, 32 * 4
    jmp .loop_vectorized

.loop_vectorized_done:

    mov r4, nq
    shl r4, 2
    add r4, r5

ALIGN 16
.loop_vectorized_tail:
    cmp dstq, r4
    jae .loop_vectorized_tail_done

    movss xmm0, [aq]
    addss xmm0, [bq]
    movss [dstq], xmm0

    add aq, 4
    add bq, 4
    add dstq, 4
    jmp .loop_vectorized_tail

.loop_vectorized_tail_done:

.end:
    vmovups ymm7, [rsp + 32]
    vmovups ymm6, [rsp + 0]
    add rsp, 64

    vzeroupper
    RET