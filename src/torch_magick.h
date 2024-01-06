#pragma once
#include <torch/torch.h>
#include <Magick++.h>
#include <optional>

// Convert between imagemagick images and pytorch tensors

enum TensorImageType {
    TensorImageType_GRAYSCALE = 1,
    TensorImageType_RGB = 3,
    TensorImageType_RGBA = 4,
};

torch::Tensor imageToTensor(const Magick::Image& image, TensorImageType type = TensorImageType_RGB, bool seperate_channels = true);
Magick::Image tensorToImage(const torch::Tensor& tensor, bool seperate_channels = true);

std::optional<torch::Tensor> imagesToTensor(std::vector<Magick::Image> images, TensorImageType type = TensorImageType_RGB, bool seperate_channels = true);
std::vector<Magick::Image> tensorToImages(torch::Tensor tensor, bool seperate_channels = true);
