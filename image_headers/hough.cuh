#ifndef HOUGH_CUH
#define HOUGH_CUH


__global__ void hough_kernel(int* line_matrix, int* image, int width, int height, int diag);

void hough(int* line_matrix, int* image, int width, int height, int diag, unsigned int threads_per_block);

#endif