#pragma once
#include <string>
#include <vector>
#include <torch/torch.h>

std::string new_file_path(const std::string& directory, const std::string& prefix, const std::string& extension);
std::vector<uint8_t> read_file(const std::string& filename);
void write_file(const std::string& filename, const void* data, int length);
std::vector<std::string> get_paths(const std::string& path);

std::string shapeToString(c10::IntArrayRef shape);

torch::Tensor tiled_scale(
    torch::Tensor images,
    std::function<torch::Tensor(torch::Tensor)> upscale_func,
    int tile_x = 64,
    int tile_y = 64,
    int overlap = 8,
    int upscale_amount = 4,
    std::function<void(float)> progress = {}
);