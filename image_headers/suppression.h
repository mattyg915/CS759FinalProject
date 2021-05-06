#ifndef SUPPRESSION_H
#define SUPPRESSION_H
#include <vector>
#include <cstddef>

void suppression(unsigned char* image, unsigned char* output, size_t width, size_t height, const float *gradient, const float *theta);

#endif