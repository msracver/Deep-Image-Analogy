

# Deep Image Analogy

The major contributors of this repository include [Jing Liao](https://liaojing.github.io/html/index.html), [Yuan Yao](http://yuanyao.info/), [Lu Yuan](http://www.lyuan.org/), [Gang Hua](http://www.ganghua.org/) and [Sing Bing Kang](http://www.singbingkang.com/publications/) at Microsoft Research.

## Introduction

**Deep Image Analogy** is a technique to find semantically-meaningful dense correspondences between two input images. It adapts the notion of image analogy with features extracted from a Deep Convolutional Neural Network.

**Deep Image Analogy** is initially described in a [SIGGRAPH 2017 paper](https://arxiv.org/abs/1705.01088)


![image](https://github.com/msracver/Deep-Image-Analogy/blob/master/windows/deep_image_analogy/example/readme/teaser.png)


## Disclaimer

This is an official C++ combined with CUDA implementation of [Deep Image Analogy](https://arxiv.org/abs/1705.01088). It is worth noticing that:
- Our codes are based on [Caffe](https://github.com/Microsoft/caffe).
- Our codes only have been tested on Windows 10, Windows Server 2012 R2 and Ubuntu with CUDA 8 or 7.5.
- Our codes only have been tested on several Nvidia GPU: Titan X, Titan Z, K40, GTX770.
- The size of input image is limited, mostly should not be large than 700x500 if you use 1.0 for parameter **ratio**.


## License

© Microsoft, 2017. Licensed under an  BSD 2-Clause license.

## Citation
If you find **Deep Image Analogy** (include deep patchmatch) helpful for your research, please consider citing:
```
  @article{liao2017visual,
    title={Visual Attribute Transfer through Deep Image Analogy},
    author={Liao, Jing and Yao, Yuan and Yuan, Lu and Hua, Gang and Kang, Sing Bing},
    journal={arXiv preprint arXiv:1705.01088},
    year={2017}
  }
```

## Application

### Photo to Style

One major application of our code is to transfer the style from a painting to a photo.
<div>
<img src="https://github.com/msracver/Deep-Image-Analogy/blob/master/windows/deep_image_analogy/example/readme/p2s1.png"/>
<img src="https://github.com/msracver/Deep-Image-Analogy/blob/master/windows/deep_image_analogy/example/readme/p2s2.png"/>
</div>

### Style to Style

It can also swap the styles between two artworks.

![image](https://github.com/msracver/Deep-Image-Analogy/blob/master/windows/deep_image_analogy/example/readme/s2s.png)

### Style to Photo

The most challenging application is converting a sketch or a painting to a photo.

<img src = "https://github.com/msracver/Deep-Image-Analogy/blob/master/windows/deep_image_analogy/example/readme/s2p3.png">

<img src = "https://github.com/msracver/Deep-Image-Analogy/blob/master/windows/deep_image_analogy/example/readme/s2p4.png">

### Photo to Photo

It can do color transfer between two photos, such as generating time lapse.

![image](https://github.com/msracver/Deep-Image-Analogy/blob/master/windows/deep_image_analogy/example/readme/p2p.png)

## Getting Started

### Prerequisite

- Linux or Mac OS X
- CUDA 8 or 7.5

### Configure & Build

- Configuration for building Caffe. Just follow t he tutorial from [Caffe](http://caffe.berkeleyvision.org/).
- Use configuration script by typing ```sh scripts/config_deep_image_analogy.sh``` under root folder.
- Modify the CUDA path in ```Makefile.config.example``` and rename it to ```Makefile.config```.
- Compile Caffe, make sure you installed all the dependencies. Just type ```make all```.
- Add libraries built by Caffe into ```LD_LIBRARY_PATH``` by ```export LD_LIBRARY_PATH="./build/lib"```.
- Compile deep_image_analogy by ```sh scripts/make_deep_image_analogy.sh```.

### Demo

Open ```main.cpp``` in ```examples/deep_image_analogy/source/``` to see how to run a demo. You need to set several parameters which have been mentioned in the paper. To be more specific, you need to set

- **path_model**, where the VGG-19 model is.
- **path_A**, the input image A.
- **path_BP**, the input image BP.
- **path_output**, the output path.
- **GPU Number**, GPU ID you want to run this experiment.
- **Ratio**, the ratio to resize the inputs before sending them into the network.
- **Blend Weight**, the level of weights in blending process.
- **Flag of WLS Filter**, if you are trying to do photo style transfer, we recommend to switch this on to keep the structure of original photo.

To run the demo, just type:
```
./deep_image_analogy examples/deep_image_analogy/models/ examples/deep_image_analogy/demo/content.png examples/deep_image_analogy/demo/style.png examples/deep_image_analogy/demo/output/ 0 0.5 2 0
```

### Tips

- We often test images of size 600x400 and 448x448.
- We set ratio to 1.0 by default. Specifically, for face (portrait) cases, we find ratio = 0.5 often make the results better.
- Blend weight controls the result appearance. If you want the result to be more like original content photo, please increase it; if you want the result more faithful to the style, please reduce it.
- For the four applications, our settings are mostly (but not definitely):
  - Photo to Style: blend weight=3, ratio=0.5 for face and ratio=1 for other cases.
  - Style to Style: blend weight=3, ratio=1.
  - Style to Photo: blend weight=2, ratio=0.5.
  - Photo to Photo: blend weight=3, ratio=1.

## Acknowledgments

Our codes acknowledge [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [PatchMatch](http://gfx.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php), [CudaLBFGS](https://github.com/jwetzl/CudaLBFGS) and [Caffe](https://github.com/BVLC/caffe). We also acknowledge to the authors of our image and style examples but we do not own the copyrights of them.
