#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH


__global__ convolve_kernel(unsigned char* image, unsigned char* output, int width, int height, const float* mask, int m);

void convolve(unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m);

#endif //CONVOLUTION_CUH
