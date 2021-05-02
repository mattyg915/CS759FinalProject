#include <vector>
#include <iostream>
#include <cstdlib>
#include "../image_headers/stb_image.h"
#include "../image_headers/image_utils.cuh"

bool load_image(std::vector<unsigned char>& image, const char* filename, int& width, int& height, int& features, int force_features)
{
    // ... force_features = # 8-bit components per pixel ...
    // ... 'features' will always be the number that it would have been if you set force_features to 0
    unsigned char* data = stbi_load(filename, &x, &y, &features, force_features);
    if (data != nullptr)
    {
        image = std::vector<unsigned char>(data, data + x * y * force_features);
    }
    stbi_image_free(data);
    return (data != nullptr);
}

__global__ void rgb_to_greyscale_kernel(unsigned char* orig_image, unsigned char* output)
{
    int num_channels = 3;

    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int index = num_channels * (output_index);

    double r = orig_image[index];
    double g = orig_image[index + 1];
    double b = orig_image[index + 2];

    double grey = (0.299 * r) + (0.587 * g) + (0.114 * b);

    output[output_index] = grey;
}

/**
 * Takes an image with 3 channels, RGB, and converts to single channel greyscale
 * @param width width in elements of the image array
 * @param height height in elements of the image array
 * @param orig_image original image array
 * @param output array to output to
 */
void rgb_to_greyscale(int width, int height, std::vector<unsigned char>& image, unsigned char* output)
{
    int num_channels = 3;
    int input_size = width * height * num_channels;
    int output_size = width * height;
    int threads_per_block = 256;

    int num_blocks = (output_size - 1) / threads_per_block + 1;

    // copy data to the device
    unsigned char *dImage, *dOutput;
    cudaMalloc((void **)&dImage, input_size * sizeof(unsigned char));
    cudaMalloc((void **)&dOutput, output_size * sizeof(unsigned char));
    cudaMemcpy(dImage, image, input_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, output, output_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // event timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    rgb_to_greyscale_kernel<<<num_blocks, threads_per_block>>>(dImage, dOutput);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float numMs;
    cudaEventElapsedTime(&numMs, start, stop);

    std::cout << "to greyscale in cuda took " << numMs << "ms" << std::endl;

    // copy back
    cudaMemcpy(output, dOutput, output_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}