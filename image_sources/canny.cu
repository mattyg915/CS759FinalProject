#include "../image_headers/canny.cuh"
#include "../image_headers/threshold.cuh"
#include "../image_headers/convolution.cuh"
#include "../image_headers/gradient.cuh"
#include "../image_headers/suppression.cuh"
#include "../image_headers/image_utils.cuh"
#include "../image_headers/hystersis.cuh"
#include <vector>
#include <iostream>

void canny(uint8_t* image, uint8_t* output, float* theta, float* gradient, uint8_t* I_x, uint8_t* I_y, size_t width, size_t height) {

    float gaussian_blur_kernel[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
    float k_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float k_y[9] = {1, 2, 1, 0, 0, 0, -1, -2 , -1};

    int size = width * height;
    int threads_per_block = 256;
    int num_blocks = (size - 1) / threads_per_block + 1;

    convolve_kernel<<<num_blocks, threads_per_block>>>(image, output, width, height, gaussian_blur_kernel, 3);
    convolve_kernel<<<num_blocks, threads_per_block>>>(image, I_x, width, height, k_x, 3);
    convolve_kernel<<<num_blocks, threads_per_block>>>(image, I_y, width, height, k_y, 3);

    gradient_kernel<<<num_blocks, threads_per_block>>>(I_x, I_y, gradient, width);
    angle_kernel<<<num_blocks, threads_per_block>>>(I_x, I_y, theta, width);

    suppression_kernel<<<num_blocks, threads_per_block>>>(image, output, width, height, gradient, theta);

    threshold_kernel<<<num_blocks, threads_per_block>>>(output, image, width, height, 50, 200, 255, 25);

    hystersis_kernel<<<num_blocks, threads_per_block>>>(image, output, width, height, 50, 200, 255, 25);

}