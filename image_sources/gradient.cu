#include "../image_headers/gradient.cuh"
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>

_global__ void gradient_kernel(const float* I_x, const float* I_y, float* output, size_t width)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = output_index % width;
    int y = output_index / width;

    float x = (float) I_x[i*width + j];
    float y = (float) I_y[i*width + j];
    float root = sqrt(x * x + y * y);
    output[output_index] = root;
}

_global__ void angle_kernel(const float* I_x, const float* I_y, float* output, size_t width)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = output_index % width;
    int y = output_index / width;

    float x = (float) I_x[i*width + j];
    float y = (float) I_y[i*width + j];
    output[i*width + j] = atan2(y, x);
}

void generate_gradient(uint8_t* I_x, uint8_t* I_y, float* output, size_t width)
{
    int size = width * height;
    int threads_per_block = 256;
    int num_blocks = (size - 1) / threads_per_block + 1;

    // copy data to the device
    unsigned char *dIx, *dIy, *dOutput;
    cudaMalloc((void **)&dIx, size * sizeof(unsigned char));
    cudaMalloc((void **)&dIy, size * sizeof(unsigned char));
    cudaMalloc((void **)&dOutput, size * sizeof(unsigned char));
    cudaMemcpy(dImage, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, output, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    gradient_kernel<<<num_blocks, threads_per_block>>>(dIx, dIy, dOutput, width);
    angle_kernel<<<num_blocks, threads_per_block>>>(dIx, dIy, dOutput, width);

    // copy back
    cudaMemcpy(output, dOutput, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

void generate_theta(uint8_t* I_x, uint8_t* I_y, float* output, size_t width)
{
    int size = width * height;
    int threads_per_block = 256;
    int num_blocks = (size - 1) / threads_per_block + 1;

    // copy data to the device
    unsigned char *dIx, *dIy, *dOutput;
    cudaMalloc((void **)&dIx, size * sizeof(unsigned char));
    cudaMalloc((void **)&dIy, size * sizeof(unsigned char));
    cudaMalloc((void **)&dOutput, size * sizeof(unsigned char));
    cudaMemcpy(dImage, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, output, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    angle_kernel<<<num_blocks, threads_per_block>>>(dIx, dIy, dOutput, width);

    // copy back
    cudaMemcpy(output, dOutput, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}