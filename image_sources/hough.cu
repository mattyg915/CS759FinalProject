#include "../image_headers/hough.cuh"
#include <iostream>
#include <cmath>

__global__ void hough_kernel(int* line_matrix, int* image, int width, int height, int diag) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i = idx / width;
	int j = idx % width;

	if (idx < width * height) {
		for (int r = -1 * diag; r < diag; r++) {
			for (int theta = 0; theta < 360; theta++) {
				if (r == (int)(i * cos(theta) + j * sin(theta))) {
					atomicAdd(line_matrix[(r + diag) * 360 + theta]);
				}
			}
		}
	}
}

void hough(int* line_matrix, int* image, int width, int height, int diag, unsigned int threads_per_block) {
	size_t number_of_blocks = (width * height + threads_per_block - 1) / threads_per_block;
	hough_kernel<<<number_of_blocks, threads_per_block>>>(line_matrix, image, width, height, diag);
	cudaDeviceSynchronize();
}