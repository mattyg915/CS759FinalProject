#ifndef IMAGE_UTILS_CUH
#define IMAGE_UTILS_CUH

bool load_image(std::vector<unsigned char>& image, const char* filename, int& x, int&y, int& features, int force_features);
__global__ void rgb_to_greyscale(int width, int height, std::vector<unsigned char>& orig_image, unsigned char* output);

#endif //IMAGE_UTILS_CUH
