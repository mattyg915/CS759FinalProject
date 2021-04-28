#ifndef GRADIENT_H
#define GRADIENT_H
#include <vector>
#include <cstddef>

void generate_gradient(uint8_t* I_x, uint8_t* I_y, float* output, size_t width, size_t height);
void generate_theta(uint8_t* I_x, uint8_t* I_y, float* output, size_t width, size_t height);

#endif