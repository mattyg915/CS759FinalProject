#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include <vector>
#include <cstddef>

void convolve(unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m);

#endif
