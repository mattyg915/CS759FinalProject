#include "../image_headers/threshold.h"
#include <vector>

void threshold_hystersis(uint8_t* image, uint8_t* output, size_t width, size_t height, float low_threshold, float high_threshold, int strong, int weak) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (image[i*width + j] > high_threshold) {
                output[i*width + j] = strong;
            }
            else if (image[i*width + j] > low_threshold) {
                output[i*width + j] = weak;
            } else {
                output[i*width + j] = 0;
            }
        }
    }

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            if ((output[i*width + j]) > 0 && (output[i*width + j] < strong)) {
                for (int k = -1; k < 2; k++) {
                    for (int l = -1; l<2; l++) {
                        if (output[(i+k)*width + (j+l)] == strong) {
                            output[i*width + j] = strong;
                            k = 2;
                            l = 2;
                        }
                    }
                }
                if (output[i*width + j] < strong) {
                    output[i * width + j] = 0;
                }
            }
        }
    }
}