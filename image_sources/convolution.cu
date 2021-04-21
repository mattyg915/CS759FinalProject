#include "../image_headers/convolution.cuh"
#include <iostream>

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

    output[output_index] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float result = calcFx(image, x + i - m / 2, y + j - m / 2, width, height);
            output[output_index] += mask[i * m + j] * result;
            if (output_index == ((height * width) / 2))
            {
                printf("blockIdx.x = %d | blockDim.x = %d | threadIdx.x = %d\n", blockIdx.x, blockDim.x, threadIdx.x);
                printf("x = %d | y = %d | i = %d | j = %d | calcFx[i] = %d | calcFx[j] = %d\n", x, y, i, j, x + i - m / 2, y + j - m / 2);
                printf("result is %f\n", result);
            }
        }
    }
}

void convolve(const unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m)
{
    int size = width * height;
    int maskSize = m * m;
    int threads_per_block = height;
    int num_blocks = width;

    // copy data to the device
    unsigned char *dImage, *dOutput;
    float *dMask;
    cudaMalloc((void **)&dImage, size * sizeof(unsigned char));
    cudaMalloc((void **)&dOutput, size * sizeof(unsigned char));
    cudaMalloc((void **)&dMask, maskSize * sizeof(float));
    cudaMemcpy(dImage, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, output, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, mask, maskSize * sizeof(float), cudaMemcpyHostToDevice);


    convolve_kernel<<<num_blocks, threads_per_block>>>(dImage, dOutput, width, height, dMask, m);
    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(output, dOutput, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}