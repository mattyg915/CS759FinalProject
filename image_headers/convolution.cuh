#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

__global__ void convolve_kernel(const unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m);
__global__ void convolve_kernel2(const unsigned char* image, float* output, int width, int height, const float *mask, int m);
void convolve(const unsigned char* image, unsigned char* output, int width, int height, const float *mask, int m);

#endif //CONVOLUTION_CUH
