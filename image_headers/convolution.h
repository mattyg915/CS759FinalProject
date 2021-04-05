#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include <vector>
#include <cstddef>

void convolve(uint8_t* image, uint8_t* output, size_t width, size_t height, const float *mask, size_t m);

#endif
