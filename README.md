# CS759FinalProject
CS759 Final Project - Image Processing

Test image library: compile code with command `g++ main.cpp image_sources/image_utils.cpp -Wall -O3 -std=c++14 -o main`.
Run with `./main {filepath to image}`. It will should output a greyscale jpg of the original input image.

Compile command for parallelized image filter: `nvcc image_filter.cu image_sources/convolution.cu image_sources/image_utils.cpp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o image_filter_cu`