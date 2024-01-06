import torch
from safetensors.torch import save_file

model = torch.load('4x-UltraSharp.pth')
save_file(model, '4x-UltraSharp.safetensors')
print('Model converted to safetensors format successfully.')
