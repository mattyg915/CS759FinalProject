#include "../image_headers/convolution.h"

float calcFx(const unsigned char* image, int i, int j, int width, int height) {
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

void convolve(const unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int output_index = y * width + x;
            float accumulator = 0;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    float result = calcFx(image, x + i - m / 2, y + j - m / 2, width, height);
                    float accumulatee = mask[i * m + j] * result;
                    accumulator += accumulatee;
                }
            }
            output[output_index] = accumulator;
        }
    }
}