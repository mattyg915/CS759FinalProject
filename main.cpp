#include <iostream>
#include <vector>
#include <fstream>
#include "image_headers/image_utils.h"

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include "image_headers/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "image_headers/stbi_image_write.h"
}

int main(int argc, char* argv[])
{
    #define CHANNEL_NUM 3

    // Must have exactly 1 command line argument
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

    std::cout << "Image width = " << width << '\n';
    std::cout << "Image height = " << height << '\n';

    auto* pixels = new uint8_t[width * height * CHANNEL_NUM];

    rgb_to_greyscale(width, height, image, pixels);

    stbi_write_jpg("output.jpg", width, height, 1, pixels, 100);

    return 0;
}
