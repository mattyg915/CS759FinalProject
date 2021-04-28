#include "../image_headers/suppression.h"
#include <vector>
#include <iostream>

void suppression(uint8_t* image, uint8_t* output, size_t width, size_t height, const float *gradient, const float *theta) {
    const float pi = 3.14159265358979323846;
    int count = 0;
    for (int x=1; x<height - 1; x++) {
        for (int y=1; y<width - 1; y++) {

            int q = 255;
            int r = 255;

            float angle = theta[x * height + y];
            angle = (angle * 180) / pi;

            if (( (0 <= angle) && (angle < 22.5)) || ((157.5 <= angle) && (angle <= 180))){
                q = image[x*width + (y+1)];
                r = image[x*width + (y - 1)];
            }
            else if ((22.5 <= angle) && (angle < 67.5)) {
                q = image[(x+1)*width + (y-1)];
                r = image[(x-1)*width + (y+1)];
            } else if ((67.5 <= angle) && (angle < 112.5)) {
                q = image[(x+1)*width + (y)];
                r = image[(x-1)*width + (y)];
            } else if ((112.5 <= angle) && (angle < 157.5)){
                q = image[(x-1)*width + (y-1)];
                r = image[(x+1)*width + (y+1)];
            }

            if ( (image[x*width + y] >= q) && (image[x*width + y] >= r) ) {
                count = count + 1;
                output[x*width + y] = image[x*width + y];
            } else {
                output[x*width + y] = 0;
            }


        }
    }
    printf("%d\n", count);
}