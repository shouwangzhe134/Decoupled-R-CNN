import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

@HEADS.register_module()
class DoubleDecoupledRefineRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """
    Decoupled and refine RoI head
    """

    def __init__(self,
                num_stages, 
                stage_loss_weights,
                reg_roi_scale_factor,
                bbox_roi_extractor=None,
                bbox_head=None,
                mask_roi_extractor=None,
                mask_head=None,
                shared_head=None,
                train_cfg=None,
                test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, 'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.reg_roi_scale_factor = reg_roi_scale_factor
        super(DoubleDecoupledRefineRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """
        Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor:
            bbox_head:
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head= nn.ModuleList()

        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor for _ in range(self.num_stages)]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages

        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))
    
    def init_mask_head(self, mask_roi_extractor, mask_head):
        """
        Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor:
            mask_head:
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
    
    def init_assigner_sampler(self):
        """
        Initaliza assigner and sampler for each stage.
        """
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for i, rcnn_train_cfg in enumerate(self.train_cfg):
                if i == 0:
                    self.bbox_assigner.append(
                        build_assigner(rcnn_train_cfg.assigner)
                    )
                    bbox_cls_sampler = build_sampler(rcnn_train_cfg.sampler_cls)
                    bbox_reg_sampler = build_sampler(rcnn_train_cfg.sampler_reg)
                    self.bbox_sampler.append((bbox_cls_sampler, bbox_reg_sampler))
                else:
                    self.bbox_assigner.append(
                        build_assigner(rcnn_train_cfg.assigner)
                    )
                    self.bbox_sampler.append(build_sampler(rcnn_train_cfg.sampler))

    def init_weights(self, pretrained):
        """
        Initialize the weights in head

        Args:
            pretrained:
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.shared_head:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()
    
    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                if i ==0:
                    outs = outs + (bbox_results['cls_score'],
                                bbox_results['bbox_pred'])
                else:
                    outs = outs + (bbox_results['bbox_pred'],)
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'], )
        return outs
    
    def _bbox_forward(self, stage, x, rois):
        """
        Box head forward function used in both training and testing time.
        """
        if stage == 0:
            bbox_roi_extractor = self.bbox_roi_extractor[stage]
            bbox_head = self.bbox_head[stage]
            bbox_cls_feats= bbox_roi_extractor(
                x[:bbox_roi_extractor.num_inputs], rois
            )
            bbox_reg_feats = bbox_roi_extractor(
                x[:bbox_roi_extractor.num_inputs],
                rois,
                roi_scale_factor=self.reg_roi_scale_factor
            )
            cls_score, bbox_pred = bbox_head(bbox_cls_feats, bbox_reg_feats)
            bbox_results = dict(
                cls_score=cls_score,
                bbox_pred=bbox_pred,
                bbox_feats=bbox_cls_feats
            )
            return bbox_results
        else:
            bbox_roi_extractor = self.bbox_roi_extractor[stage]
            bbox_head = self.bbox_head[stage]
            bbox_feats = bbox_roi_extractor(
                x[:bbox_roi_extractor.num_inputs],
                rois,
                roi_scale_factor=self.reg_roi_scale_factor
            )
            bbox_pred = bbox_head(bbox_feats)
            bbox_results = dict(bbox_pred=bbox_pred, bbox_feats=bbox_feats)
            return bbox_results


    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        if stage == 0:
            rois = bbox2roi([torch.cat((res[0].bboxes, res[1].bboxes), dim=0) for res in sampling_results])
            bbox_results = self._bbox_forward(stage, x, rois)
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
            loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                                bbox_results['bbox_pred'], rois,
                                                *bbox_targets)

            bbox_results.update(
                loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
            return bbox_results
        else:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_results = self._bbox_forward(stage, x, rois)
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
            loss_bbox = self.bbox_head[stage].loss(bbox_results['bbox_pred'], rois,
                                                *bbox_targets)

            bbox_results.update(
                loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
            return bbox_results

    def forward_train(self,
                        x,
                        img_metas,
                        proposal_list,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None,
                        gt_mask=None):
        """
        Args:
            x:
            img_metas:
            proposals:
            gt_bboxes:
            gt_labels:
            gt_bboxes_ignore:
            gt_mask:
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            if i == 0:
                sampling_results = []
                if self.with_bbox or self.with_mask:
                    bbox_assigner = self.bbox_assigner[i]
                    bbox_sampler = self.bbox_sampler[i]
                    num_imgs = len(img_metas)
                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    for j in range(num_imgs):
                        assign_result = bbox_assigner.assign(
                            proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                            gt_labels[j]
                        )
                        sampling_cls_result = bbox_sampler[0].sample(
                            assign_result[0],
                            proposal_list[j][0],
                            gt_bboxes[j],
                            gt_labels[j],
                            feats=[lvl_feat[j][None] for lvl_feat in x]
                        )
                        sampling_reg_result = bbox_sampler[1].sample(
                            assign_result[1],
                            proposal_list[j][1],
                            gt_bboxes[j],
                            gt_labels[j],
                            feats=[lvl_feat[j][None] for lvl_feat in x]
                        )
                        sampling_results.append((sampling_cls_result, sampling_reg_result))
                
                bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels, rcnn_train_cfg)
                for name, value in bbox_results['loss_bbox'].items():
                    losses[f's{i}.{name}'] = (value * lw if 'loss' in name else value)
                
                # refine bboxes
                if i < self.num_stages -1:
                    #pos_is_gts = [torch.cat([res[0].pos_is_gt, res[1].pos_is_gt], dim=0) for res in sampling_results]
                    # bbox_targets is a tuple
                    #roi_labels = bbox_results['bbox_targets'][0]
                    #labels_cls = bbox_results['bbox_targets'][0]
                    labels_reg = bbox_results['bbox_targets'][2]
                    with torch.no_grad():
                        roi_labels = torch.where(
                            (labels_reg == self.bbox_head[i].num_classes) | (labels_reg < 0),
                            bbox_results['cls_score'][:, :-1].argmax(1),
                            labels_reg
                        )
                        proposal_list = self.bbox_head[i].refine_pos_bboxes(
                            bbox_results['rois'], roi_labels, labels_reg,
                            bbox_results['bbox_pred'], img_metas
                        )
            
            else:
                sampling_results = []
                if self.with_bbox or self.with_mask:
                    bbox_assigner = self.bbox_assigner[i]
                    bbox_sampler = self.bbox_sampler[i]
                    num_imgs = len(img_metas)
                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    
                    for j in range(num_imgs):
                        assign_result = bbox_assigner.assign(
                            proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                            gt_labels[j]
                        )
                        sampling_result = bbox_sampler.sample(
                            assign_result,
                            proposal_list[j],
                            gt_bboxes[j],
                            gt_labels[j],
                            feats=[lvl_feat[j][None] for lvl_feat in x]
                        )
                        sampling_results.append(sampling_result)
                bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        rcnn_train_cfg)
                for name, value in bbox_results['loss_bbox'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value
                    )
                
                # refine bboxes
                if i < self.num_stages - 1:
                    labels_reg = bbox_results['bbox_targets'][0]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_pos_bboxes(
                            bbox_results['rois'], None, labels_reg,
                            bbox_results['bbox_pred'], img_metas
                        )

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """
        Test without augmentation.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi_stage
        #ms_bbox_result = {}
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            # split batch bbox prediction back to each image
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            if 'cls_score' in bbox_results:
                cls_score = bbox_results['cls_score']
                cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_results['bbox_pred']
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            if i == 0:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(
                        rois[j], bbox_label[j], bbox_pred[j], img_metas[j]
                    )
                    for j in range(num_imgs)
                ])
            elif i < self.num_stages - 1:
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(
                        rois[j], None, bbox_pred[j], img_metas[j]
                    )
                    for j in range(num_imgs)
                ])
        # apply bbox post_processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg
            )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]

        return bbox_results
    
    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """
        Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the 
        scale of img[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                    scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-scale
            #ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                if i == 0:
                    cls_score = bbox_results['cls_score']
                    bbox_label = bbox_results['cls_score'][:, :-1].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'], img_meta[0]
                    )
            
                elif i < self.num_stages - 1:
                    rois = self.bbox_head[i].regress_by_class(
                        rois, None, bbox_results['bbox_pred'], img_meta[0]
                    )
            
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None
            )
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg
        )
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img
        )

        bbox_result = bbox2result(det_bboxes, det_labels,
                                    self.bbox_head[-1].num_classes)
        
        return bbox_result
        

