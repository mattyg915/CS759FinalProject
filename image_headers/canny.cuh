#ifndef CANNY_CUH
#define CANNY_CUH
#include <vector>
#include <cstddef>

void canny(unsigned char* image, unsigned char* output, float* theta, float* gradient, float* I_x, float* I_y, size_t width, size_t height);

#endif