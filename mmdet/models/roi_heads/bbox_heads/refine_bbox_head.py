import torch
import torch.nn as nn

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply)
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from mmdet.models.losses import accuracy


@HEADS.register_module()
class RefineBBoxHead(BBoxHead):
    """
    2FC layers for refine regression
    refine head used in Decoupled and Refine R-CNN
    """
    def __init__(self,
                num_reg_fcs=2,
                fc_out_channels=1024,
                with_cls=False,
                *args,
                **kwargs):
        super(RefineBBoxHead, self).__init__(with_cls=False, *args, **kwargs)
        self.num_reg_fcs = num_reg_fcs
        self.fc_out_channels = fc_out_channels

        self.reg_fcs = self._add_fc_branch(self.num_reg_fcs)

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.fc_out_channels, out_dim_reg)
        self.relu = nn.ReLU(inplace=True)

    def _add_fc_branch(self, num_fcs):
        """
        Add the fc branch which consists of a sequential of fc layers.
        """
        in_channels = self.in_channels
        if not self.with_avg_pool:
            in_channels *= self.roi_feat_area
        branch_fcs = nn.ModuleList()
        for i in range(num_fcs):
            fc_in_chnnels = (in_channels if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_chnnels, self.fc_out_channels))
        return branch_fcs
    
    def __init_weights(self):
        nn.init.normal_(self.fc_reg.weight, 0, 0.01)
        nn.init.constant_(self.fc_reg.bias, 0)
        for m in self.reg_fcs.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant(m.bias, 0)
    
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        for reg_fc in self.reg_fcs:
            x = self.relu(reg_fc(x))
        bbox_pred = self.fc_reg(x)

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
                # labels_reg.size(0) = 1024, batch-size=2
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    #avg_factor=4/3*labels.size(0),
                    #avg_factor=labels.size(0), # labels_reg.size(0) = 1024
                    avg_factor=2*labels.size(0), # avg_factor = 2048 (1024 per batch) lamda=0.5
                    reduction_override=reduction_override
                )
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
            return losses
