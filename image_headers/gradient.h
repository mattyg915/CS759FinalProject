#ifndef GRADIENT_H
#define GRADIENT_H
#include <vector>
#include <cstddef>

void generate_gradient(unsigned char* I_x, unsigned char* I_y, float* output, size_t width, size_t height);
void generate_theta(unsigned char* I_x, unsigned char* I_y, float* output, size_t width, size_t height);

#endif