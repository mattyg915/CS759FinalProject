#include <iostream>
#include <vector>
#include <chrono>
#include "image_headers/image_utils.h"
#include "image_headers/convolution.h"

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
    auto* sharpened_output = new unsigned char[width * height];
    const float sharpen_kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    high_resolution_clock::time_point to_grey_start;
    high_resolution_clock::time_point to_grey_end;
    duration<double, std::milli> to_grey_duration;

    to_grey_start = high_resolution_clock::now();

    rgb_to_greyscale(width, height, image, pixels);

    to_grey_end = high_resolution_clock::now();
    to_grey_duration = std::chrono::duration_cast<duration<double, std::milli>>(to_grey_end - to_grey_start);
    cout  << "to greyscale took " << to_grey_duration.count() << "ms" << std::endl;

    high_resolution_clock::time_point convolve_start;
    high_resolution_clock::time_point convolve_end;
    duration<double, std::milli> convolve_duration;

    convolve_start = high_resolution_clock::now();

    convolve(pixels, sharpened_output, width, height, sharpen_kernel, 3);

    convolve_end = high_resolution_clock::now();
    convolve_duration = std::chrono::duration_cast<duration<double, std::milli>>(convolve_end - convolve_start);
    cout << "sequential convolution took " << convolve_duration.count() << "ms" << std::endl;

    stbi_write_jpg("output.jpg", width, height, 1, pixels, 100);
    stbi_write_jpg("output_sharpened_synchronous.jpg", width, height, 1, sharpened_output, 100);

    return 0;
}
