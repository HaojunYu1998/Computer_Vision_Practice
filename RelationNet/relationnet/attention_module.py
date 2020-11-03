import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Instances
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

__all__ = ["AttentionModule", "AttentionNMSModule"]


class AttentionModule(nn.Module):
    def __init__(
        self, fc_dim, emb_dim, feat_dim, dim, groups, num_reg_classes, device
    ):
        """
        Args:
            fc_dim (int): output dim of fully connected layer in attention module
            emb_dim (int): position embedding dim
            feat_dim (int): feature dim
            dim (Tuple(int)): (query_dim, key_dim, value_dim)
            group (int): multi-head numbers
        """
        super().__init__()
        ############################### parameters #################################
        self.device = device
        self.fc_dim, self.emb_dim = fc_dim, emb_dim
        self.feat_dim, self.dim, self.groups = feat_dim, dim, groups
        assert self.fc_dim == self.groups, "The fc_dim doesn't match groups!"
        assert self.dim[0] == self.dim[1], "Matrix multiply requires same dimensions!"
        assert dim[0] % groups == 0 and dim[2] % groups == 0
        self.dim_groups = (dim[0]//groups, dim[1]//groups, dim[2]//groups)	
        self.num_reg_classes = num_reg_classes
        ############################### modules ####################################
        self.pos_fc = nn.Linear(emb_dim, fc_dim).to(device)
        self.query_fc = nn.Linear(feat_dim, dim[0]).to(device)
        self.key_fc = nn.Linear(feat_dim, dim[1]).to(device)
        self.conv1x1_out = nn.Conv2d(fc_dim * feat_dim, dim[2], (1,1), groups=fc_dim).to(device)
        ############################# intialization ################################
        mean, std = 0.0, 0.01
        nn.init.normal_(self.pos_fc.weight, mean)
        nn.init.constant_(self.pos_fc.bias, mean)
        nn.init.normal_(self.query_fc.weight, mean, std)
        nn.init.constant_(self.query_fc.bias, mean)
        nn.init.normal_(self.key_fc.weight, mean, std)
        nn.init.constant_(self.key_fc.bias, mean)
        nn.init.normal_(self.conv1x1_out.weight, mean, std)
        nn.init.constant_(self.conv1x1_out.bias, mean)

    def forward(self, roi_feat, position_embedding):
        """
        Args:
            roi_feat (Tensor): (batch_images, num_boxes, feat_dim)
            position_embedding (Tensor): (batch_images, num_boxes, nongt_dim, emb_dim)
        """
        batch_images, num_boxes, nongt_dim = position_embedding.shape[:3]
        # (batch_images, nongt_dim, feat_dim)
        nongt_roi_feat = roi_feat[:,:nongt_dim,:]
        # (batch_images, num_boxes * nongt_dim, emb_dim)
        position_embedding = position_embedding.view(batch_images, num_boxes * nongt_dim, self.emb_dim)
        # (batch_images, num_boxes * nongt_dim, fc_dim)
        # => (batch_images, num_boxes, nongt_dim, fc_dim)
        # => (batch_images, num_boxes, fc_dim, nongt_dim)
        aff_weight	= F.relu(self.pos_fc(position_embedding)).view(
            batch_images, num_boxes, nongt_dim, self.fc_dim
        ).transpose(2, 3)
        #################################### multi head ####################################
        # (batch_images, num_boxes, dim[0])
        # => (batch_images, num_boxes, groups, dim[0]/groups) 
        # => (batch_images, groups, num_boxes, dim[0]/groups)
        q_data = self.query_fc(roi_feat).view(
            batch_images, num_boxes, self.groups, self.dim_groups[0]
        ).transpose(1, 2)
        # (batch_images, nongt_dim, dim[1])
        # => (batch_images, nongt_dim, groups, dim[1]/groups) 
        # => (batch_images, groups, nongt_dim, dim[1]/groups)
        k_data = self.key_fc(nongt_roi_feat).view(
            batch_images, nongt_dim, self.groups, self.dim_groups[1]
        ).transpose(1, 2)
        # (batch_images, nongt_dim, feat_dim)
        v_data = nongt_roi_feat
        # (batch_images, groups, num_boxes, dim[0]/groups) * (batch_images, groups, dim[1]/groups, nongt_dim)
        # => (batch_images, groups, num_boxes, nongt_dim)
        aff = torch.matmul(q_data, k_data.transpose(2, 3))
        # (batch_images, group, num_boxes, nongt_dim)
        # => (batch_images, num_boxes, group, nongt_dim)
        aff_scaled = (
            (1.0 / np.sqrt(self.dim_groups[1])) * aff
        ).transpose(1, 2)
        # (batch_images, num_boxes, fc_dim, nongt_dim)
        weighted_aff = torch.log(torch.max(
            aff_weight, torch.Tensor([[[1e-6]]]).to(self.device)
        )) + aff_scaled
        # (batch_images, num_boxes * fc_dim, nongt_dim)
        aff_softmax = F.softmax(
            weighted_aff, dim=3
        ).view(batch_images, num_boxes * self.fc_dim, nongt_dim)
        # (batch_images, num_boxes * fc_dim, nongt_dim) * (batch_images, nongt_dim, feat_dim)
        # => (batch_images, num_boxes * fc_dim, feat_dim)
        output_t = torch.bmm(aff_softmax, v_data)
        # (batch_images, num_boxes, fc_dim * feat_dim, 1, 1)
        output_t = output_t.view(batch_images * num_boxes, self.fc_dim * self.feat_dim, 1, 1)
        # (batch_images * num_boxes, dim[2], 1, 1)
        # => (batch_images, num_boxes, dim[2])
        output = self.conv1x1_out(
            output_t
        ).view(batch_images, num_boxes, self.dim[2])
        return output

class AttentionNMSModule(nn.Module):
    def __init__(
        self, fc_dim, emb_dim, feat_dim, dim, groups, num_reg_classes, device
    ):
        """
        Args:
            fc_dim (int): output dim of fully connected layer in attention module
            emb_dim (int): position embedding dim
            feat_dim (int): feature dim
            dim (Tuple(int)): (query_dim, key_dim, value_dim)
            group (int): multi-head numbers
        """
        super().__init__()
        ############################### parameters #################################
        self.device = device
        self.fc_dim, self.emb_dim = fc_dim, emb_dim
        self.feat_dim, self.dim, self.groups = feat_dim, dim, groups
        assert self.fc_dim == self.groups, "The fc_dim doesn't match groups!"
        assert self.dim[0] == self.dim[1], "Matrix multiply requires same dimensions!"
        assert dim[0] % groups == 0 and dim[2] % groups == 0
        self.dim_groups = (dim[0]//groups, dim[1]//groups, dim[2]//groups)
        self.num_reg_classes = num_reg_classes
        ############################### modules ####################################
        self.pos_fc = nn.Linear(emb_dim, fc_dim).to(device)
        self.query_fc = nn.Linear(feat_dim, dim[0]).to(device)
        self.key_fc = nn.Linear(feat_dim, dim[1]).to(device)
        self.conv1x1_out = nn.Conv2d(fc_dim * feat_dim, dim[2], (1,1), groups=fc_dim).to(device)
        ############################# intialization ################################
        mean, std = 0.0, 0.01
        nn.init.normal_(self.pos_fc.weight, mean)
        nn.init.constant_(self.pos_fc.bias, mean)
        nn.init.normal_(self.query_fc.weight, mean, std)
        nn.init.constant_(self.query_fc.bias, mean)
        nn.init.normal_(self.key_fc.weight, mean, std)
        nn.init.constant_(self.key_fc.bias, mean)
        nn.init.normal_(self.conv1x1_out.weight, mean, std)
        nn.init.constant_(self.conv1x1_out.bias, mean)

    def forward(self, roi_feat, position_embedding):
        """
        Args:
            roi_feat (Tensor): (batch_images, num_boxes, num_classes, feat_dim)
            position_embedding (Tensor): (batch_images, num_classes, num_boxes, num_boxes, emb_dim)
        Return:
            output (Tensor): (batch_images, num_boxes, num_classes, fc_dim)
        """
        batch_images, num_classes, num_boxes = position_embedding.shape[:3]
        # (batch_images, num_classes * num_boxes, feat_dim)
        roi_feat = roi_feat.transpose(1, 2).contiguous().view(batch_images, -1, self.feat_dim)
        # (batch_images, num_classes * num_boxes * num_boxes, emb_dim)
        position_embedding = position_embedding.view(batch_images, num_classes * num_boxes **2, self.emb_dim)
        # (batch_images, num_classes * num_boxes * num_boxes, fc_dim)
        # => (batch_images, num_classes, num_boxes, num_boxes, fc_dim)
        # => (batch_images, num_classes, fc_dim, num_boxes, num_boxes)
        # => (batch_images, num_classes * fc_dim, num_boxes, num_boxes)
        aff_weight	= F.relu(self.pos_fc(position_embedding)).view(
            batch_images, num_classes, num_boxes, num_boxes, self.fc_dim
        ).permute(0, 1, 4, 2, 3).contiguous().view(batch_images, -1, num_boxes, num_boxes)
        #################################### multi head ####################################
        assert self.dim[0] == self.dim[1], 'Matrix multi requires the same dims!'
        # (batch_images, num_classes * num_boxes, dim[0])
        # => (batch_images, num_classes, num_boxes, groups, dim[0]/groups) 
        # => (batch_images, groups * num_classes, num_boxes, dim[0]/groups)
        q_data = self.query_fc(roi_feat).view(
            batch_images, num_classes, num_boxes, self.groups, self.dim_groups[0]
        ).transpose(2, 3)
        q_data_batch = q_data.contiguous().view(
            batch_images, -1, num_boxes, self.dim_groups[0]
        )
        # (batch_images, num_classes * num_boxes, dim[1])
        # => (batch_images, num_classes, num_boxes, groups, dim[1]/groups) 
        # => (batch_images, num_classes, group, snum_boxes, dim[1]/groups) 
        k_data = self.key_fc(roi_feat).view(
            batch_images, num_classes, num_boxes, self.groups, self.dim_groups[1]
        ).transpose(2, 3)
        # => (batch_images, groups * num_classes, num_boxes, dim[1]/groups)
        k_data_batch = k_data.contiguous().view(
            batch_images, -1, num_boxes, self.dim_groups[1]
        )
        # (batch_images, num_classes * num_boxes, feat_dim)
        v_data = roi_feat.view(batch_images, num_classes, num_boxes, self.feat_dim)
        # (batch_images, groups * num_classes, num_boxes, dim[0]/groups) 
        # * (batch_images, groups * num_classes, dim[1]/groups, nongt_dim)
        # => (batch_images, groups * num_classes, num_boxes, num_boxes)
        aff = torch.matmul(q_data_batch, k_data_batch.transpose(2, 3))
        # (batch_images, group * num_classes, num_boxes, num_boxes)
        aff_scaled = 1.0 / np.sqrt(self.dim_groups[1]) * aff
        assert self.fc_dim == self.groups, "Check the dimensions in attention!"
        # (batch_images, num_classes * fc_dim, num_boxes, num_boxes)
        weighted_aff = torch.log(
            torch.max(aff_weight, torch.full(aff_weight.shape, 1e-6).to(self.device))
        ) + aff_scaled
        # (batch_images, num_classes * fc_dim, num_boxes, num_boxes)
        # => (batch_images, num_classes, fc_dim * num_boxes, num_boxes)
        aff_softmax = F.softmax(
            weighted_aff, dim=3
        ).view(batch_images, num_classes, self.fc_dim * num_boxes, num_boxes)
        # (batch_images, num_classes, num_boxes * fc_dim, num_boxes) 
        # * (batch_images, num_classes, num_boxes, feat_dim)
        # => (batch_images, num_classes, num_boxes * fc_dim, feat_dim)
        output_t = torch.matmul(aff_softmax, v_data)
        # (batch_images, num_classes, fc_dim, num_boxes, feat_dim)
        output_t = output_t.view(
            batch_images, num_classes, self.fc_dim, num_boxes, self.feat_dim
        )
        # (batch_images, fc_dim, feat_dim, num_boxes, num_classes) 
        # => (batch_images, fc_dim * feat_dim, num_boxes, num_classes)
        output_t = output_t.permute(0, 2, 4, 3, 1).contiguous().view(
            batch_images, -1, num_boxes, num_classes
        )
        # (batch_images, fc_dim * feat_dim, num_boxes, num_classes)
        # => (batch_images, dim[2], num_boxes, num_classes)
        # => (batch_images, num_boxes, num_classes, dim[2])
        output = self.conv1x1_out(
            output_t
        ).view(batch_images, self.dim[2], num_boxes, num_classes).permute(0, 2, 3, 1)
        return output, aff_softmax.view(
            batch_images, num_classes * self.fc_dim, num_boxes, num_boxes
        )