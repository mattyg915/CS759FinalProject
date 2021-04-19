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
    int x = threadIdx.x;
    int y = blockIdx.x;

    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("index is %d\n", output_index);
    output[output_index] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float result = calcFx(image, x + i - m / 2, y + j - m / 2, width, height);

            output[output_index] += mask[i * m + j] * result;
        }
    }
}

void convolve(const unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m)
{
    int size = width * height;
    int num_threads = 64;
    int num_blocks = (size - 1) / num_threads + 1;

    // copy data to the device
    unsigned char *dImage, *dOutput;
    cudaMalloc((void **)&dImage, size * sizeof(unsigned char));
    cudaMalloc((void **)&dOutput, size * sizeof(unsigned char));
    cudaMemcpy(dImage, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, output, size * sizeof(unsigned char), cudaMemcpyHostToDevice);


    convolve_kernel<<<num_blocks, num_threads>>>(dImage, dOutput, width, height, mask, m);
    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(output, dOutput, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}