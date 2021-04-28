#ifndef THRESHOLD_H
#define THRESHOLD_H
#include <vector>
#include <cstddef>

void threshold_hystersis(uint8_t* image, uint8_t* output, size_t width, size_t height, float low_threshold, float high_threshold, int strong, int weak);

#endif