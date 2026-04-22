#include <stddef.h>
#include <stdint.h>

int simdcnn_debug_overlaps_f(const float *x, size_t nx, const float *y, size_t ny)
{
    uintptr_t x_start = (uintptr_t)x;
    uintptr_t x_end = x_start + nx * sizeof(float);
    uintptr_t y_start = (uintptr_t)y;
    uintptr_t y_end = y_start + ny * sizeof(float);
    return !(x_end <= y_start || y_end <= x_start);
}