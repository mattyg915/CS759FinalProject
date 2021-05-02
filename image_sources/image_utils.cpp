#include <vector>
#include "../image_headers/stb_image.h"
#include "../image_headers/image_utils.h"

bool load_image(std::vector<unsigned char>& image, const char* filename, int& x, int& y, int& features, int force_features)
{
    // ... x = width, y = height, n = # 8-bit components per pixel ...
    // ... replace '0' with '1'..'4' to force that many components per pixel
    // ... but 'features' will always be the number that it would have been if you said 0
    unsigned char* data = stbi_load(filename, &x, &y, &features, force_features);
    if (data != nullptr)
    {
        image = std::vector<unsigned char>(data, data + x * y * force_features);
    }
    stbi_image_free(data);
    return (data != nullptr);
}

/**
 * Takes an image with 3 channels, RGB, and converts to single channel greyscale
 * @param width width in elements of the image array
 * @param height height in elements of the image array
 * @param orig_image original image array
 * @param output array to output to
 */
void rgb_to_greyscale(int width, int height, std::vector<unsigned char>& orig_image, unsigned char* output)
{
    int num_channels = 3;

    int p_index = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            size_t index = num_channels * (y * width + x);
            double r = orig_image[index];
            double g = orig_image[index + 1];
            double b = orig_image[index + 2];

            double grey = (0.299 * r) + (0.587 * g) + (0.114 * b);

            output[p_index++] = grey;
        }
    }
}