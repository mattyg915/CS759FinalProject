#include <iostream>
#include <vector>
#include "image_headers/image_utils.cuh"
#include "image_headers/convolution.cuh"

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include "image_headers/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "image_headers/stbi_image_write.h"
}

int main(int argc, char* argv[])
{
#define CHANNEL_NUM 4

    // Must have exactly 3 command line arguments
    if (argc != 2)
    {
        std::cerr << "Usage: ./main filename" << std::endl;
        exit(1);
    }

    char* filename = argv[1];

    int width, height, features;
    std::vector<unsigned char> image;
    bool image_loaded = load_image(image, filename, width, height, features, CHANNEL_NUM);

    if (!image_loaded)
    {
        std::cout << "Error loading image\n";
        exit(1);
    }

    const int size = width * height;

    unsigned char* pixels = new unsigned char[size];
    unsigned char* sharpened_output = new unsigned char[size];
    const float sharpen_kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    rgb_to_greyscale(width, height, image, pixels);

    convolve(pixels, sharpened_output, width, height, sharpen_kernel, 3);

    stbi_write_jpg("output.jpg", width, height, 1, pixels, 100);
    stbi_write_jpg("output_sharpened.jpg", width, height, 1, sharpened_output, 100);

    return 0;
}
