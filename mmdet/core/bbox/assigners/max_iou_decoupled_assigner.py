import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

@BBOX_ASSIGNERS.register_module()
class MaxIoUDecoupledAssigner(BaseAssigner):
    """
    Assign a corresponding gt bbox or background to bbox_cls and bbox_reg.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index(0-based) of assigned gt

    Args:
        pos_iou_thr:
        neg_iou_thr:
        min_pos_iou:
        gt_max_assign_all: Whether to assign all boxes with the same highest
            overlap with some gt to that gt.
        ignore_iof_thr:
        ignore_wrt_candidates:
        match_low_quality: Whether to allow low quality matches.
        gpu_assign_thr:
    """

    def __init__(self, pos_iou_thr, neg_iou_thr,
                min_pos_iou=.0,
                gt_max_assign_all=True,
                ignore_iof_thr=-1,
                ignore_wrt_candidates=True,
                match_low_quality=True,
                gpu_assign_thr=-1,
                iou_calculators=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculators =build_iou_calculator(iou_calculators)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """
        Assign gt to bboxes.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr, 
            assign it to that bbox.
        4. for each gt bbox, assign its nearest proposals (may be more than one) to it sels
        
        Args:
            bboxes:
            gt_bboxes:
            gt_bboxes_ignore:
            gt_labels:
        
        Returns:
            obj: `Assignresult`: the assign result.
        """

        bboxes_cls, bboxes_reg = bboxes

        overlaps_cls = self.iou_calculators(gt_bboxes, bboxes_cls)
        assign_cls_result = self.assign_wrt_overlaps(overlaps_cls, gt_labels)

        overlaps_reg = self.iou_calculators(gt_bboxes, bboxes_reg)
        assign_reg_result = self.assign_wrt_overlaps(overlaps_reg, gt_labels)

        return (assign_cls_result, assign_reg_result)

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """
        Assign w.r.t the overlaps of bboxes with gts.

        Args:
            overlaps:
            gt_labels:
        
        Returns:
            :obj:`AssignResult`: the assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or bboxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )
        
        # for each anchor, which gt best overlaps with it 
        # fot each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0]) & (max_overlaps < self.neg_iou_thr[1])] = 0
        
        # 3. assign positive
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. for each gt bbox, assign its nearest proposals
        if self.match_low_quality:
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
        
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False
            ).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )
