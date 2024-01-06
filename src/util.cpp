#include "util.h"
#include <filesystem>
#include <regex>
#include <fstream>

std::string new_file_path(const std::string& directory, const std::string& prefix, const std::string& extension) {
  int suffix = 0;
  std::string pattern = prefix + "(\\d+).*";
  std::regex re(pattern);
  for (auto& entry : std::filesystem::directory_iterator(directory)) {
    std::string file_name = entry.path().filename().string();
    std::smatch match;
    if (std::regex_match(file_name, match, re)) {
      int num = std::stoi(match[1]);
      suffix = std::max(suffix, num);
    }
  }
  std::string new_file_name = prefix + (std::stringstream() << std::setw(6) << std::setfill('0') << std::to_string(suffix+1)).str() + extension;
  return directory + "/" + new_file_name;
}

std::vector<uint8_t> read_file(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(size);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

void write_file(const std::string& filename, const void* data, int length) {
  std::ofstream file(filename, std::ios::binary);
  file.write(reinterpret_cast<const char*>(data), length);
  file.close();
}

std::vector<std::string> get_paths(const std::string& path) {
  std::vector<std::string> paths;
  if (std::filesystem::is_regular_file(path)) {
    paths.push_back(path);
  } else if (std::filesystem::is_directory(path)) {
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      if (std::filesystem::is_regular_file(entry.path())) {
        paths.push_back(entry.path().string());
      }
    }
  }
  return paths;
}

std::string shapeToString(c10::IntArrayRef shape) {
    std::stringstream ss;
    for(int i = 0; i < shape.size(); i++) {
        ss << shape.at(i) << " ";
    }
    return ss.str();
}

torch::Tensor tiled_scale(
    torch::Tensor images,
    std::function<torch::Tensor(torch::Tensor)> upscale_func,
    int tile_x,
    int tile_y,
    int overlap,
    int upscale_amount,
    std::function<void(float)> progress
) {
    int batch_size = images.size(0);
    int channels = images.size(1);
    int height_in = images.size(2);
    int width_in = images.size(3);
    int height_out = height_in * upscale_amount;
    int width_out = width_in * upscale_amount;
    int progress_steps = int(batch_size * std::ceil(float(height_in) / (tile_y - overlap)) * std::ceil(float(width_in) / (tile_x - overlap)));
    int current_progress = 0;
    using namespace torch::indexing;
    torch::Tensor output = torch::empty({ batch_size, channels, height_out, width_out }, torch::TensorOptions().device(images.device()).dtype(torch::kFloat32).requires_grad(false));
    for (int b = 0; b < batch_size; b++) {
        torch::Tensor image_in = images.index({Slice(b, b + 1)});
        torch::Tensor out = torch::zeros({ 1, channels, height_out, width_out }, torch::TensorOptions().device(images.device()).dtype(torch::kFloat32).requires_grad(false));
        torch::Tensor out_div = torch::zeros({ 1, channels, height_out, width_out }, torch::TensorOptions().device(images.device()).dtype(torch::kFloat32).requires_grad(false));
        for (int y = 0; y < height_in; y += tile_y - overlap) {
            for (int x = 0; x < width_in; x += tile_x - overlap) {
                torch::Tensor tile_in = image_in.index({Slice(), Slice(), Slice(y, y + tile_y), Slice(x, x + tile_x)});
                torch::Tensor upscaled_tile = upscale_func(tile_in);
                torch::Tensor mask = torch::ones_like(upscaled_tile);
                int feather = overlap * upscale_amount;
                for(int t = 0; t < feather; t++) {
                    float f = ((1.0/feather) * (t + 1));
                    mask.index_put_({Slice(), Slice(), Slice(t, t + 1), Slice()}, f * mask.index({Slice(), Slice(), Slice(t, t + 1), Slice()}));
                    mask.index_put_({Slice(), Slice(), Slice(mask.size(2) - t - 1, mask.size(2) - t), Slice()}, f *  mask.index({Slice(), Slice(), Slice(mask.size(2) - t - 1, mask.size(2) - t), Slice()}));
                    mask.index_put_({Slice(), Slice(), Slice(), Slice(t, t + 1)}, f * mask.index({Slice(), Slice(), Slice(), Slice(t, t + 1)}));
                    mask.index_put_({Slice(), Slice(), Slice(), Slice(mask.size(3) - t - 1, mask.size(3) - t)}, f * mask.index({Slice(), Slice(), Slice(), Slice(mask.size(3) - t - 1, mask.size(3) - t)}));
                }
                std::initializer_list<at::indexing::TensorIndex> out_indecies = {Slice(), Slice(), Slice(y * upscale_amount, (y + tile_y) * upscale_amount), Slice(x * upscale_amount, (x + tile_x) * upscale_amount)};
                out.index_put_(out_indecies, out.index(out_indecies) + upscaled_tile * mask);
                out_div.index_put_(out_indecies, out_div.index(out_indecies) + mask);
                current_progress += 1;
                if(progress) {
                    progress(float(current_progress) / float(progress_steps));
                }
            }
        }
        output.index_put_({Slice(b, b + 1)}, out/out_div);
    }
    return output.to(images.device());
}