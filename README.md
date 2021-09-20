# Decoupled R-CNN

- This code is an official implementation of "Decoupled R-CNN: Sensitivity-Specific Detector for Higher Accurate Localization" based on the open source object detection toolbox [mmdetection](https://github.com/open-mmlab/mmdetection). 

## Introduction
Object detection is a fundamental problem in computer vision, which has been widely used in many industrial applications, such as intelligent manufacturing and intelligent video surveillance. In this work, we find that classification and regression have different sensitivities to the translation of the object, from the investigation about the availability of highly overlapping proposals. More specifically, the regression branch is more sensitive to translation than the classifier branch.  Based on it, we propose a decoupled sampling strategy for a deep detector, named Decoupled R-CNN, to decouple the proposals sampling for the two different branches, which make each top branch sensitive to translation, respectively. Furthermore, we adopt the cascaded technique for the regression branch of Decoupled R-CNN, which is an extremely simple and highly effective way of improving the performance of object detection.  Extensive empirical analyses using real-world datasets demonstrate the value of the proposed method when compared with the state-of-the-art models. Specifically, on the COCO dataset, our single model can achieve competitive performance and output highly accurate bounding boxes.

## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3 or higher
- CUDA 9.0 or higher
- GCC 5+
- mmcv

### Install mmdetection

a. Create a conda virtual environment and activate it.
```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install Pytorch and torchvision.

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

c. Install mmcv.

```shell
 pip install mmcv-full==1.1.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```

d. Clone the mmdetection repository.

```shell
git clone --branch v.2.4.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```

e. Install build requirements and then install mmdetection.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## Train and Inference
All our model is trained on 4 TITAN X GPUs with a total batch size of 8 (2 images per GPU). The learning rate is initialized as 0.01.

##### Train with a single GPU
```shell
python tools/train.py ${CONFIG_FILE}
```

##### Train with multiple GPUs
```shell
./tools/dist_train.sh ${CONFIG_FILE} 4 [optional arguments]
```

#####  Test with a single GPU

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

#####  Test with multiple GPUs

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

- CONFIG_FILE about D2Det is in [configs/decoupled_rcnn](configs/decoupled_rcnn), please refer to [getting_started.md](docs/getting_started.md) for more details.


## Results

We provide some models with different backbones and results of object detection on MS COCO dateset.

| Backbone | MS train | Lr schd | Inf time (fps) | bbox AP | Config | Download|
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| ResNet-50 | No | 2x | 6.1 | 41.5 | [config](https://github.com/shouwangzhe134/Decoupled-R-CNN/blob/main/configs/decoupled_rcnn/coco/decoupled_refine_rcnn_r50_fpn_2x_coco.py) | [model]() |
| ResNet-101 | No | 2x | 5.0 | 42.8 | [config](https://github.com/shouwangzhe134/Decoupled-R-CNN/blob/main/configs/decoupled_rcnn/coco/decoupled_refine_rcnn_r101_fpn_2x_coco.py) | [model](https://1drv.ms/u/s!Agx-U0bs-cGkm2CxPRhihGvdH6yC?e=RPzZk6) |
|ResNeXt-101-32x4d | No | 2x | 4.2 | 44.4 | [config](https://github.com/shouwangzhe134/Decoupled-R-CNN/blob/main/configs/decoupled_rcnn/coco/decoupled_refine_rcnn_x101_32x4d_fpn_2x_coco.py) | [model](https://1drv.ms/u/s!Agx-U0bs-cGkm2J9gTjkBMz_vKcF?e=VxVrTc) |
|ResNeXt-101-32x4d | Yes | 2x | 4.2 | 46.0 | [config](https://github.com/shouwangzhe134/Decoupled-R-CNN/blob/main/configs/decoupled_rcnn/coco/decoupled_refine_rcnn_x101_32x4d_fpn_mstrain_2x_coco.py) | [model](https://1drv.ms/u/s!Agx-U0bs-cGkm2FAIG9Z5G8Mzbl3?e=00JyYo) |
|ResNeXt-101_64x4d | Yes | 2x | 3.1 | 46.8 | [config](https://github.com/shouwangzhe134/Decoupled-R-CNN/blob/main/configs/decoupled_rcnn/coco/decoupled_refine_rcnn_x101_64x4d_fpn_mstrain_2x_coco.py) | [model](https://1drv.ms/u/s!Agx-U0bs-cGkm2Pj1lxlmVS6AXcl?e=BhrBoB) |

- The models based on MS training use soft-NMS at inference.


## Acknowledgement
Many thanks to the open source codes, i.e., [mmdetection](https://github.com/open-mmlab/mmdetection).
