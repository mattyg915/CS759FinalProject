#include "../image_headers/convolution.cuh"
#include <iostream>

__device__ float calcFx(const unsigned char* image, int i, int j, int width, int height) {
    int x = blockIdx.x * blockDim.x;
    int y = threadIdx.x;

    int output_index = x + y;

    float result;

    if (output_index == 512)
    {
        printf("i = %d | j = %d | width = %d | height = %d\n",i, j, width, height);
    }

    if (0 <= i && i < width && 0 <= j && j < height)
    {
        if (output_index == 512)
        {
            printf("sweet spot\n");
        }
        result = image[j * width + i];
    }

    else if ((0 <= i && i < width) || (0 <= j && j < height))
    {
        if (output_index == 512)
        {
            printf("one is in\n");
        }
        result = 1;
    }
    else
    {
        if (output_index == 512)
        {
            printf("both are out\n");
        }
        result = 0;
    }

    return result;
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
                printf("x = %d | y = %d | i = %d | j = %d | calcFx[i] = %d | calcFx[j] = %d\n", x, y, i, j, x + i - m / 2, y + j - m / 2);
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