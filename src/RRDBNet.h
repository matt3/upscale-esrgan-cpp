#pragma once
#include <torch/torch.h>
#include "TypeErasure.h"

// A C++ implementation of https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/chainner_models/architecture/RRDB.py
// RRDBNet can run inference on models for ESRGAN, BSRGAN/RealSR, Real-ESRGAN, and ESRGAN-2c2

enum NormalizationType {
    NormalizationType_None,
    NormalizationType_Batch,
    NormalizationType_Instance,
};

enum ActivationType {
    ActivationType_None,
    ActivationType_RELU,
    ActivationType_LEAKYRELU,
    ActivationType_PRELU,
};

enum Upsampler {
    Upsampler_pixelshuffle,
    Upsampler_upconv
};

enum ConvMode {
    ConvMode_CNA,
    ConvMode_NAC,
    ConvMode_CNAC,
};

class RRDBNetImpl: public torch::nn::Module {
public:
    RRDBNetImpl(
        const torch::OrderedDict<std::string, torch::Tensor>& state_dict,
        NormalizationType normalization_type=NormalizationType_None,
        ActivationType activation_type=ActivationType_LEAKYRELU,
        Upsampler upsampler=Upsampler_upconv,
        ConvMode mode=ConvMode_CNA
    );
    torch::Tensor forward(const torch::Tensor& x);
    int scale_factor() const;
private:
    TE1 _d;
};

TORCH_MODULE(RRDBNet);