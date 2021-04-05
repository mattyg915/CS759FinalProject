cmake_minimum_required(VERSION 3.17)
project(CS759FinalProject)

set(CMAKE_CXX_STANDARD 14)

add_executable(CS759FinalProject main.cpp image_headers/stb_image.h image_headers/image_utils.h image_headers/convolution.h image_sources/image_utils.cpp image_sources/convolution.cpp)