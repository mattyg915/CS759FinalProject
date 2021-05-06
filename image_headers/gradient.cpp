#include "../image_headers/gradient.h"
#include <math.h>
#include <vector>

void generate_gradient(unsigned char* I_x, unsigned char* I_y, float* output, size_t width, size_t height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float x = (float) I_x[i*width + j];
            float y = (float) I_y[i*width + j];
            output[i*width + j] = sqrt(x*x + y*y);
        }
    }
}

void generate_theta(unsigned char* I_x, unsigned char* I_y, float* output, size_t width, size_t height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float x = (float) I_x[i*width + j];
            float y = (float) I_y[i*width + j];
            output[i*width + j] = atan2(y, x);
        }
    }
}