#pragma once
#include <torch/torch.h>
#include "TypeErasure.h"

// Load and save pytorch tensors to a safetensors file.
// Uses https://github.com/syoyo/safetensors-cpp internally.

class Safetensors {
public:
    Safetensors();

    bool load_from_file(const std::string& path, bool mmap = true, std::string* warning = nullptr, std::string* error = nullptr);
    bool load_from_memory(const uint8_t *addr, const size_t nbytes, bool mmap = true, std::string* warning = nullptr, std::string* error = nullptr);
    
    bool save_to_file(const std::string& path, std::string* warning = nullptr, std::string* error = nullptr);
    bool save_to_memory(std::vector<uint8_t> *data_out, std::string* warning = nullptr, std::string* error = nullptr);

    const torch::OrderedDict<std::string, torch::Tensor>& state_dict() const;
    torch::OrderedDict<std::string, torch::Tensor>& mutable_state_dict();

    const torch::OrderedDict<std::string, std::string>& metadata() const;
    torch::OrderedDict<std::string, std::string>& mutable_metadata();

private:
    TE1 _d;
};
