#include "../image_headers/convolution.cuh"
#include <iostream>
#include <cstdlib>

__device__ float calcFx(const unsigned char* image, int i, int j, int width, int height) {
    if (0 <= i && i < width && 0 <= j && j < height)
    {
        return image[j * width + i];
    }
    else if ((0 <= i && i < width) || (0 <= j && j < height))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

__global__ void convolve_kernel(const unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = output_index % width;
    int y = output_index / width;


    float accumulator = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float result = calcFx(image, x + i - m / 2, y + j - m / 2, width, height);
            accumulator += mask[i * m + j] * result;
        }
    }

    output[output_index] = accumulator;
}

void convolve(const unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m)
{
    int size = width * height;
    int maskSize = m * m;
    int threads_per_block = 256;
    int num_blocks = (size - 1) / threads_per_block + 1;

    // copy data to the device
    unsigned char *dImage, *dOutput;
    float *dMask;
    cudaMalloc((void **)&dImage, size * sizeof(unsigned char));
    cudaMalloc((void **)&dOutput, size * sizeof(unsigned char));
    cudaMalloc((void **)&dMask, maskSize * sizeof(float));
    cudaMemcpy(dImage, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, output, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, mask, maskSize * sizeof(float), cudaMemcpyHostToDevice);

    // event timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    convolve_kernel<<<num_blocks, threads_per_block>>>(dImage, dOutput, width, height, dMask, m);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    std::cout << numMs << std::endl;

    float numMs;
    cudaEventElapsedTime(&numMs, start, stop);

    // copy back
    cudaMemcpy(output, dOutput, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}