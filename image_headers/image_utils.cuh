#ifndef IMAGE_UTILS_CUH
#define IMAGE_UTILS_CUH

bool load_image(std::vector<unsigned char>& image, const char* filename, int& x, int&y, int& features, int force_features);
void rgb_to_greyscale(int width, int height, unsigned char* image, unsigned char* output);

#endif //IMAGE_UTILS_CUH
