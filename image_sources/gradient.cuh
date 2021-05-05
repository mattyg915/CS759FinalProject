#ifndef GRADIENT_CUH
#define GRADIENT_CUH
#include <vector>
#include <cstddef>

__global__ void gradient_kernel(uint8_t* I_x, uint8_t* I_y, float* output, size_t width, size_t height);
__global__ void angle_kernel(uint8_t* I_x, uint8_t* I_y, float* output, size_t width, size_t height);

#endif