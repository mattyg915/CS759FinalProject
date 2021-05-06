#include <iostream>
#include <vector>
#include <chrono>
#include "image_headers/image_utils.h"
#include "image_headers/convolution.h"
#include "image_headers/canny.cuh"

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
    float* I_x = new float[width * height];
    float* I_y = new float[width * height];
    float* gradient = new float[width * height];
    float* theta = new float[width * height];

    // copy data to the device
    unsigned char *dpixels, *dcanny_output;
    float *dI_x, *dI_y;
    float *dgradient, *dtheta;

    size_t size  = width * height;

    cudaMalloc((void **)&dpixels, size * sizeof(unsigned char));
    cudaMalloc((void **)&dcanny_output, size * sizeof(unsigned char));
    
    cudaMalloc((void **)&dI_x, size * sizeof(float));
    cudaMalloc((void **)&dI_y, size * sizeof(float));
    
    cudaMalloc((void **)&dgradient, size * sizeof(float));
    cudaMalloc((void **)&dtheta, size * sizeof(float));

    cudaMemcpy(dpixels, pixels, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dcanny_output, canny_output, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dI_x, I_x, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dI_y, I_y, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dgradient, gradient, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dtheta, theta, size * sizeof(float), cudaMemcpyHostToDevice);

    rgb_to_greyscale(width, height, image, pixels);

    // generate timing variables
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	// timing
	cudaEventRecord(startEvent, 0);

	canny(pixels, canny_output, theta, gradient, I_x, I_y, width, height);

	// timing
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	std::cout << elapsedTime << "\n";

    // copy back
    cudaMemcpy(canny_output, dcanny_output, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg("canny.jpg", width, height, 1, canny_output, 100);

    // Free device global memory
    cudaFree(dpixels);  cudaFree(dcanny_output);  cudaFree(dI_x);

    // Free device global memory
    cudaFree(dI_y);  cudaFree(dgradient);  cudaFree(dtheta);

    return 0;
}