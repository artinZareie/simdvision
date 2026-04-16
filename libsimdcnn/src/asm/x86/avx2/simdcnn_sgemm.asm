[BITS 64]
%define public_prefix simdcnn_asm
%define private_prefix simdcnn_asm
%include "x86inc.asm"

INIT_YMM avx2

section .text
; simdcnn_asm_sgemm_avx2(dst, alpha, a, b, m, n, p)
; dst: pointer
; a  : pointer
; n  : size_t
; dst = dst + 
ALIGN 16
cvisible sgemm, 7, 9, 16, dst, alpha, a, b, m, n, p, end_ptr, orig_dst
    test nq, nq
    jz .end


.end:

    vzeroupper
    RET