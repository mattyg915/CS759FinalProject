#include <iostream>
#include <vector>
#include "image_headers/image_utils.h"
#include "image_headers/convolution.cuh"

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include "image_headers/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "image_headers/stbi_image_write.h"
}

int main(int argc, char* argv[])
{
#define CHANNEL_NUM 3

    // Must have exactly 3 command line arguments
    if (argc != 2)
    {
        std::cerr << "Usage: ./main filename num_blocks num_threads" << std::endl;
        exit(1);
    }

    char* filename = argv[1];
    int num_blocks = atoi(argv[2]);
    int num_threads = atoi(argv[3]);

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

    auto* pixels = new uint8_t[width * height];
    auto* sharpened_output = new uint8_t[width * height];
    auto* gaussian_blurred_output = new uint8_t[width * height];
    auto* edge_detect_output = new uint8_t[width * height];
    float sharpen_kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    float gaussian_blur_kernel[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
    float edge_detect_kernel[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};

    rgb_to_greyscale(width, height, image, pixels);

    convolve(pixels, sharpened_output, width, height, sharpen_kernel, 3);
    convolve(pixels, gaussian_blurred_output, width, height, gaussian_blur_kernel, 3);
    convolve(pixels, edge_detect_output, width, height, edge_detect_kernel, 3);

    stbi_write_jpg("output.jpg", width, height, 1, pixels, 100);
    stbi_write_jpg("output_sharpened.jpg", width, height, 1, sharpened_output, 100);
    stbi_write_jpg("output_blurred.jpg", width, height, 1, gaussian_blurred_output, 100);
    stbi_write_jpg("output_edge_detect.jpg", width, height, 1, edge_detect_output, 100);

    return 0;
}
