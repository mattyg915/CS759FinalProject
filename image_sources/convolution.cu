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
    int x = blockIdx.x * blockDim.x;
    int y = threadIdx.x;

    int output_index = x + y;

    output[output_index] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float result = calcFx(image, x + i - m / 2, y + j - m / 2, width, height);
            output[output_index] += mask[i * m + j] * result;
            if (output_index == 512)
            {
                printf("mask[%d] is %f\n", i * m + j, mask[i * m + j]);
            }
        }
    }
}

void convolve(const unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m)
{
    int size = width * height;
    int maskSize = m * m;
    int num_threads = 64;
    int num_blocks = (size - 1) / num_threads + 1;

    // copy data to the device
    unsigned char *dImage, *dOutput;
    float *dMask;
    cudaMalloc((void **)&dImage, size * sizeof(unsigned char));
    cudaMalloc((void **)&dOutput, size * sizeof(unsigned char));
    cudaMalloc((void **)&dMask, maskSize * sizeof(float));
    cudaMemcpy(dImage, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, output, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, mask, maskSize * sizeof(float), cudaMemcpyHostToDevice);


    convolve_kernel<<<num_blocks, num_threads>>>(dImage, dOutput, width, height, dMask, m);
    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(output, dOutput, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}