#ifndef GRADIENT_CUH
#define GRADIENT_CUH
#include <vector>
#include <cstddef>

__global__ void gradient_kernel(const float* I_x, const float* I_y, float* output, size_t width);
__global__ void angle_kernel(const float* I_x, const float* I_y, float* output, size_t width);

#endif