#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "image_headers/image_utils.h"
#include "image_headers/hough.cuh"

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
	// Must have exactly 1 command line argument
	if (argc != 3)
	{
		std::cerr << "Usage: ./hough_test numlines dimension" << std::endl;
		exit(1);
	}

	// take command line input (number of lines to display)
	int numlines = atoi(argv[1]);

	// take command line input (dimension of input image)
	int width = atoi(argv[2]);
	int height = width;

	// initialize an array with mostly black but a couple of white pixels
	auto* pixels = new uint8_t[width * height];
	for (int i = 0; i < width*height; i++) {
		if (i / width > width / 3 && i / width < 2 * width / 3 && i % width > width / 3 && i % width < 2 * width / 3) {
			pixels[i] = 255;
		}
		else{
			pixels[i] = 0;
		}
	}
	//pixels[510] = 255;
	//pixels[2125] = 255;
	//pixels[12175] = 255;
	//pixels[8025] = 255;
	//pixels[5678] = 255;

	// make a copy of pixels on the device
	int *intpixels = new int[width * height];
	for (int i = 0; i < width*height; i++) {
		intpixels[i] = (int)pixels[i];
	}
	int *dpixels;
	cudaMalloc((void**)&dpixels, width * height * sizeof(int));
	cudaMemcpy(dpixels, intpixels, width * height * sizeof(int), cudaMemcpyHostToDevice);


	// save input image
	std::string s_input = "gpu_dimension_" + std::to_string(width) + "_hough_input.jpg";
	int n_input = s_input.length();
	char s_char_input[n_input+1];
	strcpy(s_char_input, s_input.c_str());
	stbi_write_jpg(s_char_input, width, height, 1, pixels, 100);

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


	// SELF: Update this section to declare a 2d array for the (r, theta) pairs and increment in parallel
	int *line_matrix, *dline_matrix;
	line_matrix = (int*)malloc((1 + max_r) * 360 * sizeof(int));
	cudaMalloc((void**)&dline_matrix, (1 + max_r) * 360 * sizeof(int));

	// Populate line_matrix with zeros
	for (int i = 0; i < (1 + max_r) * 360; i++) {
		line_matrix[i] = 0;
	}

	// Copy line_matrix to device
	cudaMemcpy(dline_matrix, line_matrix, (1 + max_r) * 360 * sizeof(int), cudaMemcpyHostToDevice);

	// generate timing variables
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	// timing
	cudaEventRecord(startEvent, 0);

	// Call kernel to accumulate counts in dline_matrix
	hough(dline_matrix, dpixels, width, height, max_r, 1024);

	// timing
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	std:: cout << "hough gpu time for dimension " << width << ":\n";
	std::cout << elapsedTime << "\n";

	// Copy line_matrix back to host
	cudaMemcpy(line_matrix, dline_matrix, (1 + max_r) * 360 * sizeof(int), cudaMemcpyDeviceToHost);


	// Use updated line_matrix to compute best lines
	for (int r = 0; r <= max_r; r++) {
		for (int theta = 0; theta < 360; theta++) {
			int curr_count = line_matrix[360 * r + theta];
			if (curr_count > best_count[numlines - 1]) {
				insert(curr_count, r, theta, best_count, best_r, best_theta, numlines);
			}
		}
	}

	// update pixels with best line drawn on it
	for (int k = 0; k < numlines; k++) {
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				if (best_r[k] == (int)(i*cos(best_theta[k]) + j*sin(best_theta[k]))) {
					pixels[i * width + j] = 255;
				}
			}
		}
		// save image with all lines added
		if (k == numlines - 1) {
			std::string s = "gpu_dimension_" + std::to_string(width) + "_hough_output_with_" + std::to_string(k+1) + "_lines.jpg";
			int n = s.length();
			char s_char[n+1];
			strcpy(s_char, s.c_str());
			stbi_write_jpg(s_char, width, height, 1, pixels, 100);
		}
	}

	//stbi_write_jpg("hough_output_with_lines.jpg", width, height, 1, pixels, 100);
	// free memory
	free(line_matrix);
	cudaFree(dline_matrix);
	cudaFree(dpixels);

	return 0;
}