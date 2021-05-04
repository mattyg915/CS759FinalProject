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

void insert(int curr_count, int curr_r, int curr_theta, int* best_count, int* best_r, int* best_theta, int numlines) {
    for (int i = 0; i < numlines; i++) {
        if (curr_count > best_count[i]) {
            for (int j = numlines - 1; j > i; j--) {
                best_count[j] = best_count[j-1];
                best_r[j] = best_r[j-1];
                best_theta[j] = best_theta[j-1];
            }
            best_count[i] = curr_count;
            best_r[i] = curr_r;
            best_theta[i] = curr_theta;
            return;
        }
    }
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

    stbi_write_jpg("canny_square.jpg", width, height, 1, canny_output, 100);
    stbi_write_jpg("canny_suppression_square.jpg", width, height, 1, suppresion_output, 100);

    //////////////////////////// hough transform starts here ////////////////////////////

    // declare number of lines
    int numlines = 20;

    // declare timing variables
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // run the hough test to find the equation of the best line
    int* best_r = new int[numlines];
    int* best_theta = new int[numlines];
    int* best_count = new int[numlines];
    for (int i = 0; i < numlines; i++) {
        best_r[i] = 0;
        best_theta[i] = 0;
        best_count[i] = 0;
    }
    int max_r = (int)sqrt(width*width +  height*height);

    // begin timing
    start = high_resolution_clock::now();

    for (int r = -1 * max_r; r < max_r; r++) {
        for (int theta = 0; theta < 360; theta++) {
            int curr_count = 0;
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    if (r == (int)(i*cos(theta) + j*sin(theta))) {
                        if (canny_output[i * width + j] == 255) {
                            curr_count++;
                        }
                    }
                }
            }
            if (curr_count > best_count[numlines - 1]) {
                insert(curr_count, r, theta, best_count, best_r, best_theta, numlines);
            }
        }
    }

    // end timing
    end = high_resolution_clock::now();
    // print the time taken by scan in ms
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    std::cout << "sequential hough took " << duration_sec.count() << " ms\n";

    // update pixels with best line drawn on it
    for (int k = 0; k < numlines; k++) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                if (best_r[k] == (int)(i*cos(best_theta[k]) + j*sin(best_theta[k]))) {
                    pixels[i * width + j] = 255;
                }
            }
        }
        // save image with each line added
        std::string s = "square_hough_output_with_" + std::to_string(k+1) + "_lines.jpg";
        int n = s.length();
        char s_char[n+1];
        strcpy(s_char, s.c_str());
        stbi_write_jpg(s_char, width, height, 1, pixels, 100);
    }


    return 0;
}