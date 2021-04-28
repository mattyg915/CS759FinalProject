#include "../image_headers/canny.h"
#include "../image_headers/threshold.h"
#include "../image_headers/convolution.h"
#include "../image_headers/gradient.h"
#include "../image_headers/suppression.h"
#include "../image_headers/image_utils.h"
#include <vector>
#include <iostream>

void canny(uint8_t* image, uint8_t* output, uint8_t* second_output, float* theta, float* gradient, uint8_t* I_x, uint8_t* I_y, size_t width, size_t height) {

    float gaussian_blur_kernel[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
    float k_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float k_y[9] = {1, 2, 1, 0, 0, 0, -1, -2 , -1};

    convolve(image, output, width, height, gaussian_blur_kernel, 3);
    convolve(image, I_x, width, height, k_x, 3);
    convolve(image, I_y, width, height, k_y, 3);

    generate_gradient(I_x, I_y, gradient, width, height);
    generate_theta(I_x, I_y, theta, width, height);

    suppression(image, second_output, width, height, gradient, theta);

    threshold_hystersis(second_output, output, width, height, 50, 200, 255, 25);

}