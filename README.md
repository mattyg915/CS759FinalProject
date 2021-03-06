# CS759FinalProject
CS759 Final Project - Image Processing

Default images provided for testing: `bagelz.jpeg` and `cardinal_tetra.jpg`

Note: some warnings during compilation are expected as they originate in the open source library we used for reading/writing images. They will come from `stb_image.h` and `stbi_image_write.h`.

Compile commands for sequential image filter: `g++ image_filter.cpp image_sources/image_utils.cpp image_sources/convolution.cpp -Wall -O3 -std=c++14 -o image_filter` and parallelized image filter: `nvcc image_filter.cu image_sources/convolution.cu image_sources/image_utils.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o image_filter_cu`. Run program with `./image_filter{_cu} {filepath to image}`. It will take a photo and output one version of it in greyscale, and one version with a sharpening kernel applied to it.

Test Canny image: compile code with command `g++ canny_test.cpp image_sources/image_utils.cpp image_sources/canny.cpp image_sources/convolution.cpp image_sources/gradient.cpp image_sources/suppression.cpp image_sources/threshold.cpp -Wall -O3 -std=c++14 -o canny_test`. Run with `./canny_test {filepath to image}`. It should output a canny edge detection jpg of the original input image as well as the resulting jpeg after only non-maximum suppression is performed.

Test canny parallel: compile code with command `nvcc canny_test.cu image_sources/convolution.cu image_sources/canny.cu image_sources/gradient.cu image_sources/threshold.cu image_sources/hystersis.cu image_sources/suppression.cu image_sources/image_utils.cpp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 --expt-relaxed-constexpr -o canny_test_cu` Run with `cuda-memcheck ./canny_test_cu {filepath}`

Test sequential hough transformation: compile code with command `g++ hough_test.cpp image_sources/image_utils.cpp -Wall -O3 -std=c++14 -o hough_test_cpu`. Run with `./hough_test_cpu {number of lines} {dimension of test image}`. This should print the time taken to run the hough transformation, and output the test image generated (with dimension as specified in the second argument) and an output image with the specified number of hough lines shown.

Test parallel hough transformation: compile code with command `nvcc hough_test.cu image_sources/hough.cu image_sources/image_utils.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o hough_test_gpu` Run with `./hough_test_gpu {number of lines} {dimension of test image}`. This should print the time taken to run the hough transformation, and output the test image generated (with dimension as specified in the second argument) and an output image with the specified number of hough lines shown.

Sequential combined canny/hough: compile code with command `g++ combined.cpp image_sources/image_utils.cpp image_sources/canny.cpp image_sources/convolution.cpp image_sources/gradient.cpp image_sources/suppression.cpp image_sources/threshold.cpp -Wall -O3 -std=c++14 -o combined`. Run with `./combined {filepath to image}`. It should output a canny edge detection jpg of the original input image, the resulting jpeg after only non-maximum suppression is performed, and the original image with 1-20 hough transform lines overlaid (this is hardcoded as 20 for now, but can be updated in the script).
