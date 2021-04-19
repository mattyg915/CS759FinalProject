#include "../image_headers/convolution.cuh"
#include <iostream>

__device__ float calcFx(const unsigned char* image, int i, int j, int width, int height) {
    if (0 <= i && i < width && 0 <= j && j < height)
    {
        printf("hit69\n");
        return image[j * width + i];
        printf("420\n");
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

__global__ void convolve_kernel(unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    output[output_index] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float result = calcFx(image, x + i - m / 2, y + j - m / 2, width, height);
            printf("result is %f\n", result);
            output[output_index] += mask[i * m + j] * result;
        }
    }
}

void convolve(unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m)
{
    int num_threads = 32;
    int num_blocks = (width * height - 1) / num_threads + 1;
    convolve_kernel<<<num_blocks, num_threads>>>(image, output, width, height, mask, m);
}