# Upscale-ESRGAN-CPP

Upscale-ESRGAN-CPP is an image upscaler project that uses a CUDA version of libtorch and ImageMagick to upscale images using ESRGAN models. ESRGAN stands for Enhanced Super-Resolution Generative Adversarial Networks, and it is a deep learning technique that can produce high-quality images from low-resolution inputs.

## Installation

Download and extract the zip on the releases page. At this time the project has only been built on windows.

## Usage

To use the project, you need to have an ESRGAN model in a safetensor format, and an image or a directory of images to upscale. You can download ESRGAN models from online sources, such as [OpenModelDB](https://openmodeldb.info/?q=esrgan) or the [upscale wiki](https://upscale.wiki/w/index.php?title=Model_Database&oldid=1571), and use a Python script to convert them from the pickletensor format to the safetensors format as needed.

To run the project, use the following command:

```
.\enhance.exe -m .\model.safetensors -i .\input_dir\ -o .\output_dir\
```


where `model.safetensors` is the path to the upscaling model, `input_dir` is the path to the image or directory of images to upscale, and `output_dir` is the path to the directory to save the upscaled images.

You can also use the following options to customize the upscaling process:

- `-d, --disable-cuda`: Use the CPU instead of an NVIDIA GPU (very slow)
- `-f, --format arg`: Format to save the images in (default: jpg)
- `-n, --no-tile`: Disable tiling / upscale the entire image at once
- `--tile-size arg`: Set tile size for upscaling (default: 512)
- `--tile-overlap arg`: Set tile overlap for upscaling (default: 32)
- `-h, --help`: Print help

For example, to upscale an image using the CPU and save it as a png file, use the following command:

```
.\enhance.exe -m .\model.safetensors -i .\image.jpg -o .\output_dir\ -d -f png
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

 - The ESRGAN model comes from the research paper [Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang. "Esrgan: Enhanced super-resolution generative adversarial networks." Proceedings of the European Conference on Computer Vision (ECCV). 2018.](https://arxiv.org/abs/1809.00219)
 - The C++ [libtorch](https://pytorch.org/) implementation of ESRGAN in this project is based off the python implementation in [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
 - [ImageMagick](https://imagemagick.org/index.php) is used for loading and saving image files.
 - [safetensors-cpp](https://github.com/syoyo/safetensors-cpp) used for reading safetensor files.
 - [cxxopts](https://github.com/jarro2783/cxxopts) used for command line arg parsing