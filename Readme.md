### Contents
- [Installation](#installation)
- [Testing](#testing)
- [Training](#training)
- [References](#references)

## Installation

1. Install Pytorch>=1.0.0 in python3.
2. Clone this repository.
```Shell
git clone https://github.com/zhangminwen/Center-and-Scale-Prediction-CSP-Pytorch.git
```

## Testing
1. Download our pre-trained model and save in $CSP_ROOT/weights.
[model:epoch320](https://pan.baidu.com/s/19XNZDa5hJV3sFxtR_XOhlg) password:s06j
2. Test the model by using gpu.
```Shell
cd $CSP_ROOT
python ./test.py 
```
3. Test the model by using cpu.
```Shell
python ./test.py --cpu
```
The "result.jpg" is detection result.

## Training
1. Prepare the dataset in $CSP_ROOT/city. 

    The pedestrian annotations (xmin, ymin, xmax, ymax) in each image are stored in the txt document which name correspond to the image name and saved in $CSP_ROOT/city/Annotations.
    
    Images are saved in $CSP_ROOT/city/Images.
    
    Splits of training, testing, valudation are saved in $CSP_ROOT/city/ImageSets.
2. Download pretrained ResNet50 model and save in $CSP_ROOT/model.

3. Train
```Shell
  python ./train.py
```

## References

```
@inproceedings{liu2018high,
  title={High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection},
  author={Wei Liu, Shengcai Liao, Weiqiang Ren, Weidong Hu, Yinan Yu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

```

