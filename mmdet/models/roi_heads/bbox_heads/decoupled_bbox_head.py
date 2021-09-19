import torch
import torch.nn as nn

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply)
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from mmdet.models.losses import accuracy


@HEADS.register_module()
class DecoupledBBoxHead(BBoxHead):
    """
    Unshared2FC
    Bbox head used in Decoupled R-CNN
    """
    def __init__(self, 
                num_cls_fcs=2,
                num_reg_fcs=2,
                fc_out_channels=1024,
                *args,
                **kwargs):
        super(DecoupledBBoxHead, self).__init__(*args, **kwargs)
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        self.fc_out_channels = fc_out_channels

        self.cls_fcs = self._add_fc_branch(self.num_cls_fcs)
        self.reg_fcs = self._add_fc_branch(self.num_reg_fcs)

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.fc_out_channels, out_dim_reg)
        
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
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

    def init_weights(self):
        for module_list in [self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
            
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x_cls = x
        x_reg = x

        for cls_fc in self.cls_fcs:
            x_cls = self.relu(cls_fc(x_cls))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None

        for reg_fc in self.reg_fcs:
            x_reg = self.relu(reg_fc(x_reg))
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred


    def _get_cls_target_single(self, pos_bboxes, neg_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        
        return labels, label_weights, num_samples
    
    def _get_reg_target_single(self, pos_bboxes, neg_bboxes, pos_gt_labels, pos_gt_bboxes, cfg):
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
        
        return labels, bbox_targets, bbox_weights, num_samples


    def get_targets(self, sampling_results, gt_bboxes,
                    gt_labels, rcnn_train_cfg, concat=True):
        pos_bboxes_cls_list = [res[0].pos_bboxes for res in sampling_results]
        neg_bboxes_cls_list = [res[0].neg_bboxes for res in sampling_results]
        pos_gt_labels_cls_list = [res[0].pos_gt_labels for res in sampling_results]
        labels_cls, label_cls_weights, num_cls_samples = multi_apply(
            self._get_cls_target_single,
            pos_bboxes_cls_list,
            neg_bboxes_cls_list,
            pos_gt_labels_cls_list,
            cfg = rcnn_train_cfg
        )

        pos_bboxes_reg_list = [res[1].pos_bboxes for res in sampling_results]
        neg_bboxes_reg_list = [res[1].neg_bboxes for res in sampling_results]
        pos_gt_labels_reg_list = [res[1].pos_gt_labels for res in sampling_results]
        pos_gt_bboxes_reg_list = [res[1].pos_gt_bboxes for res in sampling_results]
        labels_reg, bbox_reg_targets, bbox_reg_weights, num_reg_samples = multi_apply(
            self._get_reg_target_single,
            pos_bboxes_reg_list,
            neg_bboxes_reg_list,
            pos_gt_labels_reg_list,
            pos_gt_bboxes_reg_list,
            cfg=rcnn_train_cfg
        )

        Labels_cls = []
        Label_weights = []
        Labels_reg = []
        Bbox_targets = []
        Bbox_weights = []
        tmp_pos_bboxes = sampling_results[0][0].pos_bboxes
        for i in range(len(num_cls_samples)):
            num_samples = num_cls_samples[i] + num_reg_samples[i]
            label_cls = tmp_pos_bboxes.new_full((num_samples, ),
                                        -100, 
                                        dtype=torch.long)
            label_weight = tmp_pos_bboxes.new_zeros(num_samples)
            label_reg = tmp_pos_bboxes.new_full((num_samples, ),
                                        -100, 
                                        dtype=torch.long)
            bbox_target = tmp_pos_bboxes.new_zeros(num_samples, 4)
            bbox_weight = tmp_pos_bboxes.new_zeros(num_samples, 4)

            label_cls[:num_cls_samples[i]] = labels_cls[i]
            label_weight[:num_cls_samples[i]] = label_cls_weights[i]
            
            label_reg[-num_reg_samples[i]:] = labels_reg[i]
            bbox_target[-num_reg_samples[i]:, :] = bbox_reg_targets[i]
            bbox_weight[-num_reg_samples[i]:, :] = bbox_reg_weights[i]

            Labels_cls.append(label_cls)
            Label_weights.append(label_weight)

            Labels_reg.append(label_reg)
            Bbox_targets.append(bbox_target)
            Bbox_weights.append(bbox_weight)

        if concat:
            labels_cls = torch.cat(Labels_cls, 0)
            label_weights = torch.cat(Label_weights, 0)
            labels_reg = torch.cat(Labels_reg, 0)
            bbox_targets = torch.cat(Bbox_targets, 0)
            bbox_weights = torch.cat(Bbox_weights, 0)
        return labels_cls, label_weights, labels_reg, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels_cls,
             label_weights,
             labels_reg,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels_cls,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score[labels_cls >= 0], labels_cls[labels_cls >= 0])
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels_reg >= 0) & (labels_reg < bg_class_ind)
            #with open('./debug.txt', 'a') as f:
            #    f.write('1 stage' + str(len(labels_reg >= 0)) + '\n')
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels_reg[pos_inds.type(torch.bool)]]
                # labels_reg.size(0) = 2048, batch-size=2
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    #avg_factor=4/3*0.5*labels_reg.size(0),
                    #avg_factor=0.5*labels_reg.size(0), # without refine on voc. avg_factor = 1024 (512 per batch)
                    #avg_factor=labels_reg.size(0), # without refine on coco. avg_factor = 2048 (1024 per batch)
                    avg_factor=labels_reg.size(0), # with refine. avg_factor = 2048 (1024 per batch) lamda = 0.5
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        return losses
