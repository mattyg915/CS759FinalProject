#include <iostream>
#include <vector>
#include "image_headers/image_utils.h"

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include "image_headers/stb_image.h"
}

int main(int argc, char* argv[])
{
    // Must have exactly 1 command line argument
    if (argc != 2)
    {
        std::cerr << "Usage: ./main filename" << std::endl;
        exit(1);
    }

    char* filename = argv[1];

    int width, height, features;
    std::vector<unsigned char> image;
    bool image_loaded = load_image(image, filename, width, height, features);

    if (!image_loaded)
    {
        std::cout << "Error loading image\n";
        exit(1);
    }

    std::cout << "Image width = " << width << '\n';
    std::cout << "Image height = " << height << '\n';

    return 0;
}
