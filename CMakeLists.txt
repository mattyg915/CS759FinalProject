cmake_minimum_required(VERSION 3.17)
project(CS759FinalProject)

set(CMAKE_CXX_STANDARD 14)

add_executable(CS759FinalProject image_filter.cpp image_headers/stb_image.h image_headers/image_utils.h image_headers/convolution.h image_sources/image_utils.cpp image_sources/convolution.cpp)
add_executable(CS759FinalProjectParallel main.cu image_headers/stb_image.h image_headers/image_utils.cuh image_headers/gradient.cuh image_headers/hough.cuh image_headers/convolution.cuh image_sources/hough.cu image_sources/gradient.cu image_sources/image_utils.cu image_sources/convolution.cu)