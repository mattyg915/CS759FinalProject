#ifndef SUPPRESSION_CUH
#define SUPPRESSION_CUH
#include <vector>
#include <cstddef>

__global__ void suppression_kernel(uint8_t* image, uint8_t* output, size_t width, size_t height, const float *gradient, const float *theta);

#endif