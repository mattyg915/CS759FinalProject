#include <vector>
#include "../image_headers/stb_image.h"
#include "../image_headers/image_utils.h"

bool load_image(std::vector<unsigned char>& image, const char* filename, int& x, int& y, int& features)
{
    // ... x = width, y = height, n = # 8-bit components per pixel ...
    // ... replace '0' with '1'..'4' to force that many components per pixel
    // ... but 'features' will always be the number that it would have been if you said 0
    unsigned char* data = stbi_load(filename, &x, &y, &features, 4);
    if (data != nullptr)
    {
        image = std::vector<unsigned char>(data, data + x * y * 4);
    }
    stbi_image_free(data);
    return (data != nullptr);
}