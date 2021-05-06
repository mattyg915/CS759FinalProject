#include "../image_headers/hystersis.cuh"
#include <iostream>
#include <cmath>
#include <cstdio>

__global__ void hystersis_kernel(unsigned char* image, unsigned char* output, size_t width, size_t height, float low_threshold, float high_threshold, int strong, int weak)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = output_index % width;
    int y = output_index / width;


    if ((output[output_index]) > 0 && (output[output_index] < strong)) {
        for (int k = -1; k < 2; k++) {
            for (int l = -1; l<2; l++) {
                if ((((x+k)*width + (y+l)) < width * height) && output[(x+k)*width + (y+l)] == strong) {
                    output[output_index] = strong;
                    k = 2;
                    l = 2;
                }
            }
        }
        if (output[output_index] < strong) {
            output[output_index] = 0;
        }
    }
}