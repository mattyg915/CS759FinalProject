#ifndef THRESHOLD_H
#define THRESHOLD_H
#include <vector>
#include <cstddef>

void threshold_hystersis(unsigned char* image, unsigned char* output, size_t width, size_t height, float low_threshold, float high_threshold, int strong, int weak);

#endif