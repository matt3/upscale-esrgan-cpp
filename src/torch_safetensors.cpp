#define SAFETENSORS_CPP_IMPLEMENTATION
#include <safetensors.hh>
#include "torch_safetensors.h"

namespace SafetensorsInternal {

static c10::ScalarType safeTensorDTypes[] = {
    c10::ScalarType::Bool,      //   kBOOL
    c10::ScalarType::Byte,      //   kUINT8
    c10::ScalarType::Char,      //   kINT8
    c10::ScalarType::Short,     //   kINT16
    c10::ScalarType::Short,     //   kUINT16
    c10::ScalarType::Half,      //   kFLOAT16
    c10::ScalarType::BFloat16,  //   kBFLOAT16
    c10::ScalarType::Int,       //   kINT32
    c10::ScalarType::Int,       //   kUINT32
    c10::ScalarType::Float,     //   kFLOAT32
    c10::ScalarType::Double,    //   kFLOAT64
    c10::ScalarType::Long,      //   kINT64
    c10::ScalarType::Long,      //   kUINT64
};

static safetensors::dtype torchDTypes[] = {
    safetensors::kUINT8,
    safetensors::kINT8,
    safetensors::kINT16,
    safetensors::kINT32,
    safetensors::kINT64,
    safetensors::kFLOAT16,
    safetensors::kFLOAT32,
    safetensors::kFLOAT64,
    safetensors::kBOOL,
    safetensors::kBOOL,
    safetensors::kBOOL,
    safetensors::kBOOL,
    safetensors::kBOOL,
    safetensors::kBOOL,
    safetensors::kBOOL,
    safetensors::kBOOL,
    safetensors::kBFLOAT16
};

struct SafetensorsData {
    std::unique_ptr<safetensors::safetensors_t> safetensors;
    torch::OrderedDict<std::string, torch::Tensor> state_dict;
    torch::OrderedDict<std::string, std::string> metadata;
    bool state_dict_dirty = false;
    bool metadata_dirty = false;

    void close_safetensors() {
        for(auto item : state_dict) {
            item.value().copy_(item.value());
        }
        safetensors = {};
    }

    void create_safetensors_from_dict() {
        safetensors = std::make_unique<safetensors::safetensors_t>();
        state_dict_dirty = false;
        metadata_dirty = false;

        for(auto item : metadata) {
            safetensors->metadata.insert(item.key(), item.value());
        }
        for(auto item : state_dict) {
            safetensors::tensor_t tensor;
            int dtype = int(item.value().scalar_type());
            if(dtype > sizeof(torchDTypes) / sizeof(torchDTypes[0]) || dtype < 0) {
                throw std::runtime_error(std::string("Unsupported dtype: ") + std::to_string(dtype));
            }
            tensor.dtype = torchDTypes[dtype];
            c10::IntArrayRef shape = item.value().sizes();
            tensor.shape = std::vector<size_t>(shape.begin(), shape.end());
            tensor.data_offsets[0] = safetensors->storage.size();
            tensor.data_offsets[1] = tensor.data_offsets[0] + item.value().nbytes();
            safetensors->storage.resize(safetensors->storage.size() + item.value().nbytes());
            memcpy(safetensors->storage.data() + tensor.data_offsets[0], item.value().const_data_ptr(), item.value().nbytes());
        }
    }

    bool load(const std::string& path, const uint8_t *addr, const size_t nbytes, bool file, bool mmap, std::string* warning, std::string* error) {
        safetensors = std::make_unique<safetensors::safetensors_t>();
        state_dict_dirty = false;
        metadata_dirty = false;
        state_dict.clear();
        metadata.clear();

        bool result;
        if(file) {
            if(mmap) {
                result = safetensors::mmap_from_file(path, safetensors.get(), warning, error);
            } else {
                result = safetensors::load_from_file(path, safetensors.get(), warning, error);
            }
        } else {
            if(mmap) {
                result = safetensors::mmap_from_memory(addr, nbytes, path, safetensors.get(), warning, error);
            } else {
                result = safetensors::load_from_memory(addr, nbytes, path, safetensors.get(), warning, error);
            }
        }

        if(!result) {
            safetensors = {};
            return false;
        }

        for (auto key : safetensors->tensors.keys()) {
            safetensors::tensor_t t;
            if(!safetensors->tensors.at(key, &t)) {
                throw std::runtime_error(std::string("failed to load tensor: ") + key);
            }
            uint8_t* base_addr = safetensors->mmaped ? const_cast<uint8_t*>(safetensors->databuffer_addr) : safetensors->storage.data();
            uint8_t* buf = base_addr + t.data_offsets[0];
            std::vector<int64_t> &shape = *reinterpret_cast<std::vector<int64_t>*>(&t.shape);
            if(t.dtype > sizeof(safeTensorDTypes) / sizeof(safeTensorDTypes[0]) || t.dtype < 0) {
                throw std::runtime_error(std::string("unsupported dtype: ") + std::to_string(t.dtype));
            }
            c10::ScalarType dtype = safeTensorDTypes[t.dtype];
            torch::Tensor tensor = torch::from_blob(buf, shape, torch::TensorOptions().dtype(dtype));
            state_dict.insert(key, tensor);
        }
        for (auto key : safetensors->metadata.keys()) {
            std::string value;
            if(!safetensors->metadata.at(key, &value)) {
                throw std::runtime_error(std::string("failed to load metadata: ") + key);
            }
            metadata.insert(key, value);
        }
        return true;
    }

    bool save(const std::string& path, std::vector<uint8_t> *data_out, bool file, std::string* warning, std::string* error) {
        if(safetensors && safetensors->mmaped && !metadata_dirty) {
            if(file) {
                std::ofstream ofs(path, std::ios::binary);
                ofs.write(reinterpret_cast<const char *>(safetensors->mmap_addr), safetensors->mmap_size);
                if(!ofs) {
                    return false;
                }
                return true;
            } else {
                if(!data_out) {
                    return false;
                }
                *data_out = std::vector<uint8_t>(safetensors->mmap_addr, safetensors->mmap_addr + safetensors->mmap_size);
                return true;
            }
        }

        if (safetensors && metadata_dirty) {
            close_safetensors();
        }
        if(!safetensors) {
            create_safetensors_from_dict();
        }
        
        if(file) {
            return safetensors::save_to_file(*safetensors, path, warning, error);
        } else {
            return safetensors::save_to_memory(*safetensors, data_out, warning, error);
        }
    }
};

} // namespace SafetensorsInternal

using namespace SafetensorsInternal;

Safetensors::Safetensors(): _d(TE<SafetensorsData>()){}

bool Safetensors::load_from_file(const std::string& path, bool mmap, std::string* warning, std::string* error) {
    return _d.get<SafetensorsData>().load(path, nullptr, 0, true, mmap, warning, error);
}

bool Safetensors::load_from_memory(const uint8_t *addr, const size_t nbytes, bool mmap, std::string* warning, std::string* error) {
    return _d.get<SafetensorsData>().load({}, addr, nbytes, false, mmap, warning, error);
}

bool Safetensors::save_to_file(const std::string& path, std::string* warning, std::string* error){
    return _d.get<SafetensorsData>().save(path, nullptr, true, warning, error);
}
bool Safetensors::save_to_memory(std::vector<uint8_t> *data_out, std::string* warning, std::string* error){
    return _d.get<SafetensorsData>().save({}, data_out, false, warning, error);
}

const torch::OrderedDict<std::string, torch::Tensor>& Safetensors::state_dict() const {
    return _d.get<SafetensorsData>().state_dict;
}

// If the tensor data is mmaped, a deep copy of the tensor will be made and the data unmapped before this returns.
torch::OrderedDict<std::string, torch::Tensor>& Safetensors::mutable_state_dict() {
    SafetensorsData& d = _d.get<SafetensorsData>();
    if(!d.state_dict_dirty) {
        d.close_safetensors();
        d.state_dict_dirty = true;
    }
    return d.state_dict;
}

const torch::OrderedDict<std::string, std::string>& Safetensors::metadata() const {
    return _d.get<SafetensorsData>().metadata;
}

torch::OrderedDict<std::string, std::string>& Safetensors::mutable_metadata() {
    SafetensorsData& d = _d.get<SafetensorsData>();
    d.metadata_dirty = true;
    return d.metadata;
}