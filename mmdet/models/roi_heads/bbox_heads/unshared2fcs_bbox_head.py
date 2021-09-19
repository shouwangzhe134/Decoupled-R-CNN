import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead


@HEADS.register_module()
class Unshared2FCBBoxHead(BBoxHead):
    """
    Unshared2FCs
    BBox head used in Faster R-CNN
    """
    def __init__(self, 
                num_cls_fcs=2,
                num_reg_fcs=2,
                fc_out_channels=1024,
                *args,
                **kwargs):
        super(Unshared2FCBBoxHead, self).__init__(*args, **kwargs)
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
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
        for module_list in [self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
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
