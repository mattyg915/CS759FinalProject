#include "../image_headers/threshold.cuh"
#include <iostream>
#include <cmath>
#include <cstdio>

__global__ void threshold_kernel(unsigned char* image, unsigned char* output, size_t width, size_t height, float low_threshold, float high_threshold, int strong, int weak)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = output_index % width;
    int y = output_index / width;


    if ((x > 0) && (x<height)) {
        if ((y > 0) && (y < height)) {
            if (image[output_index] > high_threshold) {
                output[output_index] = strong;
            }
            else if (image[output_index] > low_threshold) {
                output[output_index] = weak;
            } else {
                output[output_index] = 0;
            }
        }
        else {
            output[output_index] = 0;
        }
    } else {
        output[output_index] = 0;
    }
}