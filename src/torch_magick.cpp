#include "torch_magick.h"

std::optional<torch::Tensor> imagesToTensor(std::vector<Magick::Image> images, TensorImageType type, bool seperate_channels) {
    int64_t batchSize = images.size();
    if(!batchSize) {
        // Batch size must be greater than zero.
        return {};
    }
    int64_t height = images[0].rows();
    int64_t width = images[0].columns();
    int64_t channels = int64_t(type);
    for(int64_t i = 0; i < batchSize; i++) {
        if(images[i].columns() != width || images[i].rows() != height) {
            // Images must have the same dimensions and number of channels.
            return {};
        }
    }
    torch::Tensor tensor = torch::zeros({batchSize, height, width, channels}, torch::kFloat32);
    int64_t image_size = height * width * channels * sizeof(float);
    std::string map;
    if(type == TensorImageType_GRAYSCALE) {
        map = "I";
    } else if(type == TensorImageType_RGB) {
        map = "RGB";
    } else if(type == TensorImageType_RGBA) {
        map = "RGBA";
    }
    for(int64_t i = 0; i < batchSize; i++) {
        images[i].write(0, 0, width, height, map, Magick::FloatPixel, reinterpret_cast<uint8_t*>(tensor.mutable_data_ptr()) + image_size * i);
    }
    if(seperate_channels) {
        tensor = tensor.permute({0, 3, 1, 2});
    }
    return tensor;
}

torch::Tensor imageToTensor(const Magick::Image& image, TensorImageType type, bool seperate_channels) {
    return *imagesToTensor({image}, type, seperate_channels);
}

std::vector<Magick::Image> tensorToImages(torch::Tensor tensor, bool seperate_channels) {
    if(seperate_channels) {
        tensor = tensor.permute({0, 2, 3, 1});
    }
    tensor = tensor.to(torch::kFloat32).contiguous();
    int64_t batchSize = tensor.size(0);
    int64_t height = tensor.size(1);
    int64_t width = tensor.size(2);
    int64_t channels = tensor.size(3);
    std::string map;
    if(channels == TensorImageType_GRAYSCALE) {
        map = "I";
    } else if(channels == TensorImageType_RGB) {
        map = "RGB";
    } else if(channels == TensorImageType_RGBA) {
        map = "RGBA";
    }
    int64_t image_size = height * width * channels * sizeof(float);
    std::vector<Magick::Image> images = std::vector<Magick::Image>(batchSize);
    for(int64_t i = 0; i < batchSize; i++) {
        images[i].read(width, height, map, Magick::FloatPixel, reinterpret_cast<const uint8_t*>(tensor.const_data_ptr()) + image_size * i);
    }
    return images;
}

Magick::Image tensorToImage(const torch::Tensor& tensor, bool seperate_channels) {
    return tensorToImages(tensor, seperate_channels)[0];
}