import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, xavier_init

from mmdet.models.backbones.regnet import Bottleneck
from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply)
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from mmdet.models.losses import accuracy


class BasicResBlock(nn.Module):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicResBlock, self).__init__()

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@HEADS.register_module()
class DoubleRefineBBoxHead(BBoxHead):
    """
    4 conv layers for refine regression
    refine head used in Decoupled and Refine R-CNN
    """
    def __init__(self,
                num_convs=4,
                conv_out_channels=1024,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                with_cls=False,
                **kwargs
                ):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleRefineBBoxHead, self).__init__(with_cls=False, **kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        self.num_convs = num_convs
        self.conv_out_channels = conv_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels,
                                        self.conv_out_channels)
        # add conv heads
        self.conv_branch = self._add_conv_branch()

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)
    
    def _add_conv_branch(self):
        branch_convs = nn.ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
        return branch_convs
    
    def init_weights(self):
        normal_init(self.fc_reg, std=0.01)
    
    def forward(self, x):
        x_conv = self.res_block(x)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)
        
        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)
        
        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)

        return bbox_pred
    
    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_labels, pos_gt_bboxes, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        labels = pos_bboxes.new_full((num_samples, ), self.num_classes, dtype=torch.long)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes
                )
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1.0
        return labels, bbox_targets, bbox_weights
    
    def get_targets(self, sampling_results, gt_bboxes, 
                    gt_labels, rcnn_train_cfg, concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        labels, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_labels_list,
            pos_gt_bboxes_list,
            cfg=rcnn_train_cfg
        )

        if concat:
            labels = torch.cat(labels, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, bbox_targets, bbox_weights

    @force_fp32(apply_to=('bbox_pred', ))
    def loss(self,
            bbox_pred,
            rois,
            labels,
            bbox_targets,
            bbox_weights,
            reduction_override=None):
        losses = dict()
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            #with open('./debug.txt', 'a') as f:
            #    f.write(str(len(labels)) + '\n')
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4
                    )[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1, 4
                    )[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=2*labels.size(0),
                    #avg_factor=labels.size(0),
                    reduction_override=reduction_override
                )
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
            return losses
