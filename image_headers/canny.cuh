#ifndef CANNY_CUH
#define CANNY_CUH
#include <vector>
#include <cstddef>

void canny(uint8_t* image, uint8_t* output, float* theta, float* gradient, uint8_t* I_x, uint8_t* I_y, size_t width, size_t height);

#endif