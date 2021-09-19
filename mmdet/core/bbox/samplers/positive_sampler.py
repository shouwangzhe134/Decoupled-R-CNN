from abc import ABCMeta, abstractmethod

import torch

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class PositiveSampler(metaclass=ABCMeta):
    """
    positive sampler
    
    Args:
        num (int): Number of samples
    """

    def __init__(self, num, **kwargs):
        self.num = num

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """
        Randomly sample some positive samples
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)
    
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels=None, **kwargs):
        """
        Sample positive bboxes.

        Args:
            assign_result:
            bboxes:
            gt_bboxes:
            gt_labels:
        
        Returns:
            :obj:`SamplingResult`: sampling result.
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)

        num_expected_pos = int(self.num)
        pos_inds = self._sample_pos(assign_result, num_expected_pos, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of pytorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        neg_inds = self._sample_neg(assign_result, num_expected_neg, bboxes=bboxes, ** kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, 
            assign_result, gt_flags)

        return sampling_result
