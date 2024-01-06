#include <iostream>
#include <cxxopts.hpp>
#include "RRDBNet.h"
#include "torch_safetensors.h"
#include "torch_magick.h"
#include "util.h"

int main(int argc, char **argv) {
    std::string model_path;
    std::string input;
    std::string output;
    bool disable_cuda = false;
    std::string format = "jpg";
    bool no_tile = false;
    int tile_size = 512;
    int tile_overlap = 32;

    cxxopts::Options options("enhance", "Upscale images using an ESRGAN model");
    options.add_options()
        ("m,model", "Path to the upscaling model (required)", cxxopts::value<std::string>(model_path))
        ("i,input", "Path to the image or directory of images to upscale (required)", cxxopts::value<std::string>(input))
        ("o,output", "Path to the directory to save images (required)", cxxopts::value<std::string>(output))
        ("d,disable-cuda", "Use the CPU instead of an NVIDIA GPU (very slow)", cxxopts::value<bool>(disable_cuda))
        ("f,format", "Format to save the images in", cxxopts::value<std::string>(format)->default_value("jpg"))
        ("n,no-tile", "Disable tiling / upscale the entire image at once", cxxopts::value<bool>(no_tile))
        ("tile-size", "Set tile size for upscaling", cxxopts::value<int>(tile_size)->default_value("512"))
        ("tile-overlap", "Set tile overlap for upscaling", cxxopts::value<int>(tile_overlap)->default_value("32"))
        ("h,help", "Print help");
    cxxopts::ParseResult result = options.parse(argc, argv);
    
    if(result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    
    if(model_path.empty() || input.empty() || output.empty()) {
        std::cout << options.help();
        return -1;
    }

    Magick::InitializeMagick(nullptr);

    std::cout << "Loading model " << model_path << std::endl;
    torch::NoGradGuard no_grad;
    Safetensors st;
    st.load_from_file(model_path);
    RRDBNet model = RRDBNet(st.state_dict());
    if(!disable_cuda) {
        model->to(torch::kCUDA);
    }

    std::vector<std::string> paths = get_paths(input);
    std::cout << "Found " << std::to_string(paths.size()) << " files at " << input << std::endl;

    for(const std::string& path : paths) {
        std::cout << "Loading image " << path << std::endl;
        std::vector<uint8_t> image_data = read_file(path);
        Magick::Blob image_blob = Magick::Blob(image_data.data(), image_data.size());
        Magick::Image image = image_blob;
        if(!image.isValid()) {
            std::cout << "Invalid image " << path << std::endl;
            continue;
        }
        
        std::cout << "Upscaling image..." << std::endl;
        torch::Tensor imageTensor = imageToTensor(image);
        auto upscale_func = [&](torch::Tensor image){
            if(disable_cuda) {
                return model->forward(image);
            } else {
                // Do upscale on gpu and tiling on cpu
                return model->forward(image.to(torch::kCUDA)).to(torch::kCPU);
            }
        };
        torch::Tensor upscaledImageTensor;
        if(no_tile) {
            upscaledImageTensor = upscale_func(imageTensor);
        } else {
            auto progress_callback = [](float progress){
                int barWidth = 70;
                std::cout << "[";
                int pos = barWidth * progress;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress * 100.0) << " %\r";
                std::cout.flush();
            };
            upscaledImageTensor = tiled_scale(imageTensor, upscale_func, tile_size, tile_size, tile_overlap, model->scale_factor(), progress_callback);
        }
        Magick::Image upscaledImage = tensorToImage(upscaledImageTensor);
        
        std::cout << "Converting image to " << format << std::endl;
        upscaledImage.magick(format);
        Magick::Blob upscaled_blob;
        upscaledImage.write(&upscaled_blob);

        std::string filePath = new_file_path(output, "upscaled_", std::string(".") + format);
        std::cout << "Saving image to " << filePath << std::endl;
        write_file(filePath, upscaled_blob.data(), upscaled_blob.length());
    }
    
    
    std::cout << "Done!" << std::endl;

    return 0;
}
