_base_ = './decoupled_refine_rcnn_r50_fpn_2x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
