#ifndef SUPPRESSION_H
#define SUPPRESSION_H
#include <vector>
#include <cstddef>

void suppression(uint8_t* image, uint8_t* output, size_t width, size_t height, const float *gradient, const float *theta);

#endif