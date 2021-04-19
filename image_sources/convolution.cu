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
    int z = blockDim.x;
    printf("x: %d or y: %d or z: %d\n", x, y, z);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("index is %d\n",index);

    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index == 512)
    {
        printf("yes we hit 512\n");
    }
    output[output_index] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float result = calcFx(image, x + i - m / 2, y + j - m / 2, width, height);
            output[output_index] += mask[i * m + j] * result;
            if (output_index == 512)
            {
                printf("here it's %d\n", output[512]);
            }
        }
    }
}

void convolve(const unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m)
{
    int size = width * height;
    int num_threads = 64;
    int num_blocks = (size - 1) / num_threads + 1;

    // copy data to the device
    unsigned char *dImage, *dMask, *dOutput;
    cudaMalloc((void **)&dImage, size * sizeof(unsigned char));
    cudaMalloc((void **)&dOutput, size * sizeof(unsigned char));
    cudaMalloc((void **)&dMask, 9 * sizeof(unsigned char));
    cudaMemcpy(dImage, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, output, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, mask, 9 * sizeof(unsigned char), cudaMemcpyHostToDevice);


    convolve_kernel<<<num_blocks, num_threads>>>(dImage, dOutput, width, height, dMask, m);
    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(output, dOutput, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    printf(" out at 512 is %d\n", output[512]);
}