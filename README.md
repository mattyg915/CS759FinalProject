# CS759FinalProject
CS759 Final Project - Image Processing

Test image library: compile code with command `g++ main.cpp image_sources/image_utils.cpp -Wall -O3 -std=c++14 -o main`.
Run with `./main {filepath to image}`. It will should output a greyscale jpg of the original input image.

Compile command for parallelized image filter: `nvcc image_filter.cu image_sources/convolution.cu image_sources/image_utils.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o image_filter_cu`

Test Canny image: compile code with command `g++ canny_test.cpp image_sources/image_utils.cpp image_sources/canny.cpp image_sources/convolution.cpp image_sources/gradient.cpp image_sources/suppression.cpp image_sources/threshold.cpp -Wall -O3 -std=c++14 -o canny_test`. Run with `./canny_test {filepath to image}`. It should output a canny edge detection jpg of the original input image as well as the resulting jpeg after only non-maximum suppression is performed.

Sequential combined canny/hough: compile code with command `g++ combined.cpp image_sources/image_utils.cpp image_sources/canny.cpp image_sources/convolution.cpp image_sources/gradient.cpp image_sources/suppression.cpp image_sources/threshold.cpp -Wall -O3 -std=c++14 -o combined`. Run with `./combined {filepath to image}`. It should output a canny edge detection jpg of the original input image, the resulting jpeg after only non-maximum suppression is performed, and the original image with 1-20 hough transform lines overlaid (this is hardcoded as 20 for now, but can be updated in the script).
