#include "../image_headers/gradient.cuh"
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>

__global__ void gradient_kernel(const float* I_x, const float* I_y, float* output, size_t width)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = output_index % width;
    int y = output_index / width;

    float xvalue = (float) I_x[output_index];
    float yvalue = (float) I_y[output_index];
    float root = sqrt(xvalue * xvalue + yvalue * yvalue);
    output[output_index] = root;
}

__global__ void angle_kernel(const float* I_x, const float* I_y, float* output, size_t width)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = output_index % width;
    int y = output_index / width;

    float xvalue = (float) I_x[output_index];
    float yvalue = (float) I_y[output_index];
    output[output_index] = atan2(y, x);
}