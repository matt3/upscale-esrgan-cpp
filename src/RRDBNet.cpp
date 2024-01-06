#include "RRDBNet.h"
#include <torch/torch.h>
#include <iostream>
#include <regex>
#include <cmath>
#include <string>

#include <iomanip>
#include "torch_safetensors.h"
#include "torch_magick.h"

torch::nn::AnyModule act(ActivationType type, bool inplace = true, double negative_slope = 0.2, int n_prelu = 1) { 
    switch (type) {
        case ActivationType_None:
            return torch::nn::AnyModule();
        case ActivationType_RELU:
            return torch::nn::AnyModule(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(inplace)));
        case ActivationType_LEAKYRELU:
            return torch::nn::AnyModule(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(negative_slope).inplace(inplace)));
        case ActivationType_PRELU:
            return torch::nn::AnyModule(torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(n_prelu).init(negative_slope)));
        default:
            throw std::invalid_argument("Invalid activation type");
    }
}

torch::nn::AnyModule norm(NormalizationType type, int num_features) {
    switch (type) {
        case NormalizationType_None:
            return torch::nn::AnyModule();
        case NormalizationType_Batch:
            return torch::nn::AnyModule(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features)));
        case NormalizationType_Instance:
            return torch::nn::AnyModule(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(num_features)));
        default:
            throw std::invalid_argument("Invalid normalization type");
    }
}

enum PaddingType {
    PaddingType_Zero,
    PaddingType_Reflection,
    PaddingType_Replication,
};
torch::nn::AnyModule pad(PaddingType type, int padding) {
    switch (type) {
        case PaddingType_Zero:
            return torch::nn::AnyModule();
        case PaddingType_Reflection:
            return torch::nn::AnyModule(torch::nn::ReflectionPad2d(torch::nn::ReflectionPad2dOptions(padding)));
        case PaddingType_Replication:
            return torch::nn::AnyModule(torch::nn::ReplicationPad2d(torch::nn::ReplicationPad2dOptions(padding)));
        default:
            throw std::invalid_argument("Invalid padding type");
    }
}

int getValidPadding(int kernel_size, int dilation) {
    return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2;
}

class ShortcutBlockImpl : public torch::nn::Module {
public:
    ShortcutBlockImpl(const std::string& name, torch::nn::AnyModule submodule) : _submodule(submodule){
        register_module(name, _submodule.ptr());
    }
    torch::Tensor forward(torch::Tensor x) {
        return x + _submodule.forward(x);
    }
private:
    torch::nn::AnyModule _submodule;
};
TORCH_MODULE(ShortcutBlock);

struct StackSequentialImpl : torch::nn::SequentialImpl {
  using SequentialImpl::SequentialImpl;

  torch::Tensor forward(torch::Tensor x) {
    return SequentialImpl::forward(x);
  }
};
TORCH_MODULE(StackSequential);

StackSequential sequentialFromVector(const std::vector<torch::nn::AnyModule>& modules) {
    StackSequential s;
    int i = 0;
    for(const torch::nn::AnyModule &module : modules) {
        if(!module.is_empty()) {
            s->push_back(std::to_string(i++), module);
        }
    }
    // torch::OrderedDict<std::string, torch::Tensor> params = s->named_parameters();
    // for(const auto& item: params) {
    //     std::cout << item.key() << std::endl;
    // };
    // s->apply([](const std::string& key, const torch::nn::Module& module) {
    //     std::cout << key << ": " << module.name() << std::endl;
    // });
    return s;
}

std::vector<torch::nn::AnyModule> conv_block_2c2(
    int in_channels,
    int out_channels,
    ActivationType activation_type=ActivationType_RELU
){
    return {
        torch::nn::AnyModule(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 2).padding(1))),
        torch::nn::AnyModule(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 2).padding(0))),
        torch::nn::AnyModule(act(activation_type))
    };
}

std::vector<torch::nn::AnyModule> conv_block(
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride=1,
    int dilation=1,
    int groups=1,
    bool bias=true,
    PaddingType padding_type=PaddingType_Zero,
    NormalizationType normalization_type=NormalizationType_None,
    ActivationType activation_type=ActivationType_RELU,
    ConvMode mode=ConvMode_CNA,
    bool c2x2=false
){
    if(c2x2){
        return conv_block_2c2(in_channels, out_channels, activation_type);
    }
    int padding = getValidPadding(kernel_size, dilation);
    int convPadding = padding_type == PaddingType_Zero ? padding : 0;
    torch::nn::AnyModule p = pad(padding_type, padding);
    torch::nn::AnyModule c = torch::nn::AnyModule(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).
        stride(stride).
        padding(convPadding).
        dilation(dilation).
        bias(bias).
        groups(groups)));
    if(mode == ConvMode_CNA || mode == ConvMode_CNAC){
        torch::nn::AnyModule a = act(activation_type);
        torch::nn::AnyModule n = norm(normalization_type, out_channels);
        return {p, c, n, a};
    } else if(mode == ConvMode_NAC){
        bool inplace = normalization_type != NormalizationType_None || activation_type == ActivationType_None;
        torch::nn::AnyModule a = act(activation_type, inplace);
        torch::nn::AnyModule n = norm(normalization_type, in_channels);
        return {n, a, p, c};
    } else {
        throw std::invalid_argument("Invalid convolution mode");
    }
}

torch::nn::AnyModule conv1x1(int in_planes, int out_planes, int stride=1) {
    return torch::nn::AnyModule(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, 1).stride(stride).bias(false)));
}

// Residual Dense Block
// style: 5 convs
// The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
// Modified options that can be used:
//     - "Partial Convolution based Padding" arXiv:1811.11718
//     - "Spectral normalization" arXiv:1802.05957
//     - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.
//         {Rakotonirina} and A. {Rasoanaivo}
// Args:
//     nf (int): Channel number of intermediate features (num_feat).
//     gc (int): Channels for each growth (num_grow_ch: growth channel,
//         i.e. intermediate channels).
//     convtype (str): the type of convolution to use. Default: 'Conv2D'
//     gaussian_noise (bool): enable the ESRGAN+ gaussian noise (no new
//         trainable parameters)
//     plus (bool): enable the additional residual paths from ESRGAN+
//         (adds trainable parameters)
class ResidualDenseBlock_5CImpl: public torch::nn::Module {
public:
    ResidualDenseBlock_5CImpl(
        int nf,
        int kernel_size=3,
        int gc=32,
        int stride=1,
        bool bias=true,
        PaddingType padding_type=PaddingType_Zero,
        NormalizationType normalization_type=NormalizationType_None,
        ActivationType activation_type=ActivationType_RELU,
        ConvMode mode=ConvMode_CNA,
        bool plus=false,
        bool c2x2=false
    ){
        ActivationType last_activation = mode == ConvMode_CNA ? ActivationType_None : activation_type;
        int last_kernel = 3;
        if(plus) {
            _conv1x1 = conv1x1(nf, gc);
            register_module("conv1x1", _conv1x1.ptr());
        }
        _conv1 = sequentialFromVector(conv_block(nf + 0 * gc, gc, kernel_size, stride, 1, 1, bias, padding_type, normalization_type, activation_type, mode, c2x2));
        _conv2 = sequentialFromVector(conv_block(nf + 1 * gc, gc, kernel_size, stride, 1, 1, bias, padding_type, normalization_type, activation_type, mode, c2x2));
        _conv3 = sequentialFromVector(conv_block(nf + 2 * gc, gc, kernel_size, stride, 1, 1, bias, padding_type, normalization_type, activation_type, mode, c2x2));
        _conv4 = sequentialFromVector(conv_block(nf + 3 * gc, gc, kernel_size, stride, 1, 1, bias, padding_type, normalization_type, activation_type, mode, c2x2));
        _conv5 = sequentialFromVector(conv_block(nf + 4 * gc, nf, last_kernel, stride, 1, 1, bias, padding_type, normalization_type, last_activation, mode, c2x2));
        register_module("conv1", _conv1);
        register_module("conv2", _conv2);
        register_module("conv3", _conv3);
        register_module("conv4", _conv4);
        register_module("conv5", _conv5);
    }
    torch::Tensor forward(const torch::Tensor& x) {
        torch::Tensor x1 = _conv1->forward(x);
        torch::Tensor x2 = _conv2->forward(torch::cat({x, x1}, 1));
        if(!_conv1x1.is_empty()) {
            x2 += _conv1x1.forward(x);
        }
        torch::Tensor x3 = _conv3->forward(torch::cat({x, x1, x2}, 1));
        torch::Tensor x4 = _conv4->forward(torch::cat({x, x1, x2, x3}, 1));
        if(!_conv1x1.is_empty()) {
            x4 += x2;
        }
        torch::Tensor x5 = _conv5->forward(torch::cat({x, x1, x2, x3, x4}, 1));
        return x5 * 0.2 + x;
    }
private:
    torch::nn::AnyModule _conv1x1;
    StackSequential _conv1;
    StackSequential _conv2;
    StackSequential _conv3;
    StackSequential _conv4;
    StackSequential _conv5;
};
TORCH_MODULE(ResidualDenseBlock_5C);

// Residual in Residual Dense Block
// (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
class RRDBImpl: public torch::nn::Module {
public:
    RRDBImpl(
        int nf,
        int kernel_size=3,
        int gc=32,
        int stride=1,
        bool bias=true,
        PaddingType padding_type=PaddingType_Zero,
        NormalizationType normalization_type=NormalizationType_None,
        ActivationType activation_type=ActivationType_RELU,
        ConvMode mode=ConvMode_CNA,
        bool plus=false,
        bool c2x2=false,
        int index=0
    ): _rdb1(nf, kernel_size, gc, stride, bias, padding_type, normalization_type, activation_type, mode, plus,c2x2),
       _rdb2(nf, kernel_size, gc, stride, bias, padding_type, normalization_type, activation_type, mode, plus,c2x2),
       _rdb3(nf, kernel_size, gc, stride, bias, padding_type, normalization_type, activation_type, mode, plus,c2x2),
       _index(index) {
        register_module("RDB1", _rdb1);
        register_module("RDB2", _rdb2);
        register_module("RDB3", _rdb3);
    }
    torch::Tensor forward(const torch::Tensor& x) {
        torch::Tensor x1 = _rdb1->forward(x);
        torch::Tensor x2 = _rdb2->forward(x1);
        torch::Tensor x3 = _rdb3->forward(x2);
        torch::Tensor x4 = x3 * 0.2 + x;
        return x4;
    }
private:
    ResidualDenseBlock_5C _rdb1;
    ResidualDenseBlock_5C _rdb2;
    ResidualDenseBlock_5C _rdb3;
    int _index;
};
TORCH_MODULE(RRDB);

std::vector<torch::nn::AnyModule> pixelshuffle_block(
    int in_channels,
    int out_channels,
    int upscale_factor=2,
    int kernel_size=3,
    int stride=1,
    bool bias=true,
    PaddingType padding_type=PaddingType_Zero,
    NormalizationType normalization_type=NormalizationType_None,
    ActivationType activation_type=ActivationType_RELU
) {
    std::vector<torch::nn::AnyModule> conv = conv_block(in_channels, out_channels * upscale_factor * upscale_factor, kernel_size, stride, 1, 1, bias, padding_type, normalization_type, activation_type);
    conv.push_back(torch::nn::AnyModule(torch::nn::PixelShuffle(upscale_factor)));
    conv.push_back(norm(normalization_type, out_channels));
    conv.push_back(act(activation_type));
    return conv;
}

typedef c10::variant<
      torch::enumtype::kNearest,
      torch::enumtype::kLinear,
      torch::enumtype::kBilinear,
      torch::enumtype::kBicubic,
      torch::enumtype::kTrilinear>
      upsample_mode_t;

std::vector<torch::nn::AnyModule> upconv_block(
    int in_channels,
    int out_channels,
    int upscale_factor=2,
    int kernel_size=3,
    int stride=1,
    bool bias=true,
    PaddingType padding_type=PaddingType_Zero,
    NormalizationType normalization_type=NormalizationType_None,
    ActivationType activation_type=ActivationType_RELU,
    upsample_mode_t mode=torch::kNearest,
    bool c2x2=false
) {
    std::vector<torch::nn::AnyModule> s;
    std::vector<double> factor = { double(upscale_factor), double(upscale_factor) };
    s.push_back(torch::nn::AnyModule(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(factor).mode(mode))));
    std::vector<torch::nn::AnyModule> c = conv_block(in_channels, out_channels, kernel_size, stride, 1, 1, bias, padding_type, normalization_type, activation_type, ConvMode_CNA, c2x2);
    s.insert(s.end(), c.begin(), c.end());
    return s;
}

struct RRDBNetData {
    NormalizationType normalization_type;
    ActivationType activation_type;
    Upsampler upsampler;
    ConvMode mode;
    bool plus = false;
    std::map<std::string, std::string> key_mapping;
    torch::nn::AnyModule model;
    int shuffle_factor;
    int scale;
};

RRDBNetImpl::RRDBNetImpl(
    const torch::OrderedDict<std::string, torch::Tensor>& state_dict,
    NormalizationType normalization_type,
    ActivationType activation_type,
    Upsampler upsampler,
    ConvMode mode
): _d(TE<RRDBNetData>()) {
    RRDBNetData& d = _d.get<RRDBNetData>();
    
    d.normalization_type = normalization_type;
    d.activation_type = activation_type;
    d.upsampler = upsampler;
    d.mode = mode;

    std::vector<std::string> keys = state_dict.keys();
    
    for(const std::string &key: keys) {
        if(key.find("conv1x1")!= std::string::npos) {
            d.plus = true;
            break;
        }
    }

    bool old_arch = true;
    for(const std::string &key: keys) {
        if(key.find("conv_first")!= std::string::npos) {
            old_arch = false;
            break;
        }
    }

    int rrdb_modules = 0;
    for(const std::string &key: keys) {
        d.key_mapping[key] = key;
        if(key.find("RDB1.conv1.0.weight")!= std::string::npos) {
            rrdb_modules += 1;
        }
    }

    int max_upconv = 0;
    int last_layer = 0;
    std::regex pattern("model\\.(\\d+)\\.(weight|bias)");
    std::smatch match;
    for(const std::string &key: keys) {
        if(std::regex_match(key, match, pattern)) {
            int key_num = std::stoi(match[1].str());
            last_layer = std::max(last_layer, key_num);
        }
    }
    max_upconv = (last_layer - 4) / 3;

    // Create a key mapping between new and old architectures.
    if(!old_arch) {
        d.key_mapping["conv_first.weight"] = "model.0.weight";
        d.key_mapping["conv_first.bias"] = "model.0.bias";
        d.key_mapping["trunk_conv.weight"] = std::string("model.1.sub.") + std::to_string(rrdb_modules) + ".weight";
        d.key_mapping["trunk_conv.bias"] = std::string("model.1.sub.") + std::to_string(rrdb_modules) + ".bias";
        d.key_mapping["conv_body.weight"] = std::string("model.1.sub.") + std::to_string(rrdb_modules) + ".weight";
        d.key_mapping["conv_body.bias"] = std::string("model.1.sub.") + std::to_string(rrdb_modules) + ".bias";
        for(int i = 0; i < rrdb_modules; i++) {
            for(int j = 1; j <= 3; j++) {
                for(int k = 1; k <= 5; k++) {
                    d.key_mapping[std::string("RRDB_trunk.") + std::to_string(i) + ".RDB" + std::to_string(j) + ".conv" + std::to_string(k) + ".weight"] = std::string("model.1.sub.") + std::to_string(i) + ".weight";
                    d.key_mapping[std::string("RRDB_trunk.") + std::to_string(i) + ".RDB" + std::to_string(j) + ".conv" + std::to_string(k) + ".bias"] = std::string("model.1.sub.") + std::to_string(i) + ".bias";
                    d.key_mapping[std::string("body.") + std::to_string(i) + ".rdb" + std::to_string(j) + ".conv" + std::to_string(k) + ".weight"] = std::string("model.1.sub.") + std::to_string(i) + ".weight";
                    d.key_mapping[std::string("body.") + std::to_string(i) + ".rdb" + std::to_string(j) + ".conv" + std::to_string(k) + ".bias"] = std::string("model.1.sub.") + std::to_string(i) + ".bias";
                }
            }
        }

        // upconv layers
        std::regex pattern("(upconv|conv_up)(\\d)\\.(weight|bias)");
        std::smatch match;
        for(const std::string &key: keys) {
            if(std::regex_match(key, match, pattern)) {
                int key_num = std::stoi(match[2].str());
                std::string key_type = match[3].str();
                d.key_mapping[key] = std::string("model.") + std::to_string(key_num * 3) + "." + key_type;
                max_upconv = std::max(max_upconv, key_num);
            }
        }
        // final layers
        d.key_mapping["HRconv.weight"] = std::string("model.") + std::to_string(3 * max_upconv + 2) + ".weight";
        d.key_mapping["HRconv.bias"] = std::string("model.") + std::to_string(3 * max_upconv + 2) + ".bias";
        d.key_mapping["conv_last.weight"] = std::string("model.") + std::to_string(3 * max_upconv + 4) + ".weight";
        d.key_mapping["conv_last.bias"] = std::string("model.") + std::to_string(3 * max_upconv + 4) + ".bias";
    }

    std::map<std::string, std::string> inverse_key_mapping;
    for(const std::string &key: keys) {
        inverse_key_mapping[d.key_mapping[key]] = key;
    }

    c10::IntArrayRef input_shape = state_dict[inverse_key_mapping["model.0.weight"]].sizes();
    int in_nc = input_shape[1];
    int out_nc = state_dict[inverse_key_mapping[std::string("model.") + std::to_string(last_layer) + ".weight"]].size(0);
    int scale = std::pow(2, max_upconv);
    int num_filters = input_shape[0];
    bool c2x2 = false;
    if(input_shape[input_shape.size()-2] == 2) {
        c2x2 = true;
        scale = std::round(std::sqrt(double(scale) / 4));
    }

    int shuffle_factor = 0;
    if((in_nc == out_nc * 4 || in_nc == out_nc * 16) && (out_nc == in_nc / 4 || out_nc == in_nc / 16)) {
        shuffle_factor = int(std::sqrt(double(in_nc) / out_nc));
    }

    d.model = torch::nn::AnyModule([&]{
        std::vector<torch::nn::AnyModule> model = conv_block(in_nc, num_filters, 3, 1, 1, 1, true, PaddingType_Zero, NormalizationType_None, ActivationType_None, ConvMode_CNA, c2x2);
        model.push_back(torch::nn::AnyModule(ShortcutBlock("sub", [&]{
            std::vector<torch::nn::AnyModule> rrdb_blocks;
            for(int i = 0; i < rrdb_modules; i++) {
                rrdb_blocks.push_back(torch::nn::AnyModule(RRDB(num_filters, 3, 32, 1, true, PaddingType_Zero, normalization_type, activation_type, ConvMode_CNA, d.plus, c2x2, i)));
            }
            std::vector<torch::nn::AnyModule> last = conv_block(num_filters, num_filters, 3, 1, 1, 1, true, PaddingType_Zero, normalization_type, ActivationType_None, mode, c2x2);
            rrdb_blocks.insert(rrdb_blocks.end(), last.begin(), last.end());
            return torch::nn::AnyModule(sequentialFromVector(rrdb_blocks));
        }())));
        std::vector<torch::nn::AnyModule> upsampler_blocks = [&]{
            if(scale == 3) {
                if(upsampler == Upsampler_upconv) {
                    return upconv_block(num_filters, num_filters, 3, 3, 1, true, PaddingType_Zero, NormalizationType_None, d.activation_type, torch::kNearest, c2x2);
                } else {
                    return pixelshuffle_block(num_filters, num_filters, 3, 3, 1, true, PaddingType_Zero, NormalizationType_None, d.activation_type);
                }
            } else {
                std::vector<torch::nn::AnyModule> blocks;
                for(int i = 0; i < std::log2(scale); i++) {
                    if(upsampler == Upsampler_upconv) {
                        std::vector<torch::nn::AnyModule> b  = upconv_block(num_filters, num_filters, 2, 3, 1, true, PaddingType_Zero, NormalizationType_None, d.activation_type, torch::kNearest, c2x2);
                        blocks.insert(blocks.end(), b.begin(), b.end());
                    } else {
                        std::vector<torch::nn::AnyModule> b = pixelshuffle_block(num_filters, num_filters, 2, 3, 1, true, PaddingType_Zero, NormalizationType_None, d.activation_type);
                        blocks.insert(blocks.end(), b.begin(), b.end());
                    }
                }
                return blocks;
            }
        }();
        model.insert(model.end(), upsampler_blocks.begin(), upsampler_blocks.end());
        std::vector<torch::nn::AnyModule> hr_conv0 = conv_block(num_filters, num_filters, 3, 1, 1, 1, true, PaddingType_Zero, NormalizationType_None, activation_type, ConvMode_CNA, c2x2);
        model.insert(model.end(), hr_conv0.begin(), hr_conv0.end());
        std::vector<torch::nn::AnyModule> hr_conv1 = conv_block(num_filters, out_nc, 3, 1, 1, 1, true, PaddingType_Zero, NormalizationType_None, ActivationType_None, ConvMode_CNA, c2x2);
        model.insert(model.end(), hr_conv1.begin(), hr_conv1.end());
        StackSequential s = sequentialFromVector(model);
        register_module("model", s);
        return s;
    }());

    if(shuffle_factor) {
        in_nc /= shuffle_factor * shuffle_factor;
        scale /= shuffle_factor;
    }

    d.shuffle_factor = shuffle_factor;
    d.scale = scale;

    torch::OrderedDict<std::string, torch::Tensor> model_params = named_parameters();
    for(auto& item: model_params) {
        if(const torch::Tensor* t = state_dict.find(item.key())) {
            item.value().set_data(*t);
        } else {
            std::cout << "tensor not found: " << item.key() << std::endl;
        }
    };
}
torch::Tensor RRDBNetImpl::forward(const torch::Tensor& x) {
    // shape of x is [batch_size (1 for a single image), num_channels (3 for RGB), image_height, image_width]
    RRDBNetData& d = _d.get<RRDBNetData>();
    if(d.shuffle_factor) {
        int h = x.size(2);
        int w = x.size(3);
        int mod_pad_h = (d.shuffle_factor - h % d.shuffle_factor) % d.shuffle_factor;
        int mod_pad_w = (d.shuffle_factor - w % d.shuffle_factor) % d.shuffle_factor;
        torch::Tensor x1 = torch::pad(x, {0, mod_pad_w, 0, mod_pad_h}, "reflect");
        torch::Tensor x2 = torch::pixel_unshuffle(x1, d.shuffle_factor);
        torch::Tensor x3 = d.model.forward(x2);
        return x3.slice(2, 0, h * d.scale).slice(3, 0, w * d.scale);
    } else {
        return d.model.forward(x);
    }
}
int RRDBNetImpl::scale_factor() const {
    return _d.get<RRDBNetData>().scale;
}