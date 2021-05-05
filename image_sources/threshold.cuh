#ifndef THRESHOLD_CUH
#define THRESHOLD_CUH
#include <vector>
#include <cstddef>

__global__ void threshold_kernel(uint8_t* image, uint8_t* output, size_t width, size_t height, float low_threshold, float high_threshold, int strong, int weak);

#endif