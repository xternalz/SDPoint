#  Stochastic Downsampling for Cost-Adjustable Inference and Improved Regularization in Convolutional Networks (SDPoint)

![](http://oi64.tinypic.com/2ly1lk0.jpg)

This repository contains the code for the SDPoint method proposed in
> [Stochastic Downsampling for Cost-Adjustable Inference and Improved Regularization in Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kuen_Stochastic_Downsampling_for_CVPR_2018_paper.pdf)<br/>**CVPR 2018**

### Citation
If you find this code useful for your research, please cite
```
@article{kuen2018stochastic,
  title={Stochastic Downsampling for Cost-Adjustable Inference and Improved Regularization in Convolutional Networks},
  author={Kuen, Jason and Kong, Xiangfei and Zhe, Lin and Wang, Gang and Yin, Jianxiong and See, Simon and Tan, Yap-Peng},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

### Dependencies
  - Python 3
  - [PyTorch 0.4.0](https://github.com/pytorch/pytorch/tree/v0.4.0) (and torchvision)

### Dataset
Set up ImageNet dataset according to https://github.com/pytorch/examples/tree/master/imagenet.

### Supported Architectures
* ResNets - `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
* Pre-activation ResNets (PreResNets) - `preresnet18`, `preresnet34`, `preresnet50`, `preresnet101`, `preresnet152`, `preresnet200`
* ResNeXts - `resnext50`, `resnext101`, `resnext152`

### Training
```
python main.py -a resnext101 [imagenet-folder with train and val folders]
```

### Evaluation
The different SDPoint instances are evaluated one by one. For each instance, the model accumulates Batch Norm statistics from the training set. The validation results (top-1 and top-5 accuracies) and model FLOPs are saved to the file with the filename specified by `--val-results-path` [default: val_results.txt].
```
python main.py -a resnext101 --resume checkpoint.pth.tar --evaluate [imagenet-folder with train and val folders]
```
