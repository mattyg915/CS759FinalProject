#include "../image_headers/convolution.h"
#include <vector>

float calcFx(const uint8_t* image, size_t i, size_t j, std::size_t width, std::size_t height) {
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

void convolve(uint8_t* image, uint8_t* output, size_t width, size_t height, const float *mask, size_t m)
{
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            output[y * width + x] = 0;
            for (size_t i = 0; i < m; i++)
            {
                for (size_t j = 0; j < m; j++)
                {
                    float result = calcFx(image, x + i - m / 2, y + j - m / 2, width, height);
                    output[y * width + x] += mask[i * m + j] * result;
                }
            }
        }
    }
}