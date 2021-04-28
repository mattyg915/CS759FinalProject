#include <iostream>
#include <vector>
#include <chrono>
#include "image_headers/image_utils.h"
#include "image_headers/convolution.h"
#include "image_headers/canny.h"

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include "image_headers/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "image_headers/stbi_image_write.h"
}

int main(int argc, char* argv[])
{
    #define CHANNEL_NUM 3

    using std::cout;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

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

    cout << "Image width = " << width << std::endl;
    cout << "Image height = " << height << std::endl;

    auto* pixels = new unsigned char[width * height];
    auto* canny_output = new unsigned char[width * height];
    auto* suppresion_output = new unsigned char[width * height];
    auto* I_x = new uint8_t[width * height];
    auto* I_y = new uint8_t[width * height];
    float* gradient = new float[width * height];
    float* theta = new float[width * height];
    const float sharpen_kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    rgb_to_greyscale(width, height, image, pixels);

    high_resolution_clock::time_point canny_start;
    high_resolution_clock::time_point canny_end;
    duration<double, std::milli> canny_duration;

    canny_start = high_resolution_clock::now();

    canny(pixels, canny_output, suppresion_output, theta, gradient, I_x, I_y, width, height);

    canny_end = high_resolution_clock::now();
    canny_duration = std::chrono::duration_cast<duration<double, std::milli>>(canny_end - canny_start);
    cout << "convolve took " << canny_duration.count() << "ms" << std::endl;

    stbi_write_jpg("canny.jpg", width, height, 1, canny_output, 100);
    stbi_write_jpg("canny_suppression.jpg", width, height, 1, suppresion_output, 100);

    return 0;
}