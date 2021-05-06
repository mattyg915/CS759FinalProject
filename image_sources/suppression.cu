#include "../image_headers/suppression.cuh"
#include <iostream>
#include <cmath>
#include <cstdio>

__global__ void suppression_kernel(unsigned char* image, unsigned char* output, size_t width, size_t height, const float *gradient, const float *theta)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = output_index % width;
    int y = output_index / width;

    float pi = 3.14159265358979;

    if ((x > 0) && (x<height)) {
        if ((y > 0) && (y < width) && (((x+1)*width + y + 1) < width*height)) {
            int q = 255;
            int r = 255;

            float angle = theta[x * width + y];
            angle = (angle * 180) / pi;

            if (( (0 <= angle) && (angle < 22.5)) || ((157.5 <= angle) && (angle <= 180))){
                q = image[x*width + (y+1)];
                r = image[x*width + (y-1)];
            }
            else if ((22.5 <= angle) && (angle < 67.5)) {
                q = image[(x+1)*width + (y-1)];
                r = image[(x-1)*width + (y+1)];
            } else if ((67.5 <= angle) && (angle < 112.5)) {
                q = image[(x+1)*width + (y)];
                r = image[(x-1)*width + (y)];
            } else if ((112.5 <= angle) && (angle < 157.5)){
                q = image[(x-1)*width + (y-1)];
                r = image[(x+1)*width + (y+1)];
            }

            if ( (image[x*width + y] >= q) && (image[x*width + y] >= r) ) {
                output[x*width + y] = image[x*width + y];
            } else {
                output[x*width + y] = 0;
            }
        }
    }
}