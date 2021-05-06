#ifndef CANNY_H
#define CANNY_H
#include <vector>
#include <cstddef>

void canny(unsigned char* image, unsigned char* output, unsigned char* second_output, float* theta, float* gradient, unsigned char* I_x, unsigned char* I_y, size_t width, size_t height);

#endif