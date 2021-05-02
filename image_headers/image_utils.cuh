#ifndef IMAGE_UTILS_CUH
#define IMAGE_UTILS_CUH

bool load_image(std::vector<unsigned char>& image, const char* filename, int& width, int& height, int& features, int force_features);
void rgb_to_greyscale(int width, int height, std::vector<unsigned char>& image, unsigned char* output);

#endif //IMAGE_UTILS_CUH
