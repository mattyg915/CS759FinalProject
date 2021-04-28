#ifndef CANNY_H
#define CANNY_H
#include <vector>
#include <cstddef>

void canny(uint8_t* image, uint8_t* output, uint8_t* second_output, float* theta, float* gradient, uint8_t* I_x, uint8_t* I_y, size_t width, size_t height);

#endif