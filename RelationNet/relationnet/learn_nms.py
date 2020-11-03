import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.box_regression import Box2BoxTransform

from .attention_module import AttentionNMSModule
# from .padding import padding_tensor
from .embedding import (
    extract_rank_embedding, 
    extract_multi_position_matrix,
    extract_pairwise_multi_position_embedding,
)

__all__ = ["NMSLossCls"]

class LearnNMSModule(nn.Module):
    def __init__(self, cfg):
        super(LearnNMSModule, self).__init__()
        ############################### parameters #################################
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.iou_thresh = cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS
        self.num_thresh = len(self.iou_thresh)
        self.att_fc_dim = cfg.MODEL.RELATIONNET.ATT_FC_DIM
        self.nms_fc_dim = cfg.MODEL.RELATIONNET.NMS_FC_DIM
        self.feat_dim = cfg.MODEL.RELATIONNET.FEAT_DIM
        self.pos_emb_dim = cfg.MODEL.RELATIONNET.POS_EMB_DIM
        self.att_dim = cfg.MODEL.RELATIONNET.ATT_DIM
        self.first_n = cfg.MODEL.RELATIONNET.FIRST_N_TEST
        if self.training:
            self.first_n = cfg.MODEL.RELATIONNET.FIRST_N_TRAIN
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.num_reg_classes = 2 if cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG else self.num_classes
        self.groups = cfg.MODEL.RELATIONNET.ATT_GROUPS
        self.nms_thresh_test = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        assert self.nms_thresh_test in self.iou_thresh
        self.nms_pos_scale = cfg.MODEL.RELATIONNET.NMS_POS_SCALE
        self.nms_loss_scale = cfg.MODEL.RELATIONNET.NMS_LOSS_SCALE
        self.nms_eps = 1e-8
        ############################### modules ####################################
        self.roi_emb_fc = nn.Linear(self.feat_dim, self.nms_fc_dim).to(self.device)
        self.rank_emb_fc = nn.Linear(self.feat_dim, self.nms_fc_dim).to(self.device)
        self.logit_fc = nn.Linear(self.nms_fc_dim, self.num_thresh).to(self.device)
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.attention_module_nms_multi_head = self._build_attention_nms_module_multi_head()
        ############################# intialization ################################
        mean, std = 0.0, 0.01
        nn.init.normal_(self.roi_emb_fc.weight, mean, std)
        nn.init.constant_(self.roi_emb_fc.bias, mean)
        nn.init.normal_(self.rank_emb_fc.weight, mean, std)
        nn.init.constant_(self.rank_emb_fc.bias, mean)
        nn.init.normal_(self.logit_fc.weight, mean, std)
        nn.init.constant_(self.logit_fc.bias, -3.0)

    def _build_attention_nms_module_multi_head(self):
        attention_module_nms_multi_head = AttentionNMSModule(
            fc_dim=self.att_fc_dim, 
            feat_dim=self.nms_fc_dim, emb_dim=self.pos_emb_dim,
            dim=(self.att_dim[0], self.att_dim[1], self.nms_fc_dim), 
            groups=self.groups, num_reg_classes=self.num_reg_classes, 
            device=self.device
        )
        return attention_module_nms_multi_head
    
    @torch.no_grad()
    def get_multi_target(self, bbox, targets, scores):
        """
        Args:
            bbox (Tensor): (batch_images, first_n, num_classes, 4)

            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes (Tensor): (num_gt, 4) the bounding box of each instance.
                - gt_classes (Tensor): (num_gt,) the label for each instance with a category 
                  ranging in [0, num_classes)
            scores (Tensor): (batch_images, first_n, num_classes)
        Return:
            multi_target (Tensor): (batch_images, first_n, num_classes, num_thresh)
            pred_instances (List(Instances))
        """
        batch_images = bbox.shape[0]
        output_list = []
        for batch_idx in range(batch_images):
            output_list_per_image = []
            for cls_idx in range(self.num_classes):
                valid_gt_mask = targets[batch_idx].gt_classes == cls_idx
                valid_gt_box = targets[batch_idx].gt_boxes[valid_gt_mask, :]
                num_valid_gt = len(valid_gt_box)
                if num_valid_gt == 0:
                    # (first_n, 1, num_thresh)
                    output_list_per_image.append(
                        torch.zeros((self.first_n, 1, self.num_thresh)).to(self.device)
                    )
                else:
                    bbox_per_class = bbox[batch_idx, :, cls_idx, :]
                    score_per_class = scores[batch_idx, :, cls_idx:cls_idx+1]
                    # (first_n, num_valid_gt)
                    overlap_mat = pairwise_iou(
                        valid_gt_box, Boxes(bbox_per_class.view(-1, 4))
                    ).transpose(0, 1)
                    eye_matrix = torch.eye(num_valid_gt).to(self.device)
                    output_list_per_class = []
                    for thresh in self.iou_thresh:
                        # following mAP metric (first_n, num_valid_gt)
                        overlap_mask = (overlap_mat > thresh)
                        # When only condition is provided, this function is a shorthand 
                        # for np.asarray(condition).nonzero()
                        # (frist_n,)
                        valid_bbox_indices = torch.where(overlap_mask)[0]
                        # require score be 2-dim
                        # (first_n,) => (first_n, num_valid_gt)
                        overlap_score = score_per_class.repeat(1, num_valid_gt)
                        # first, pick the overlap_score that greater than threshold
                        overlap_score = overlap_score.mul(overlap_mask)
                        # for every proposal bbox, pick the best match gt
                        max_overlap_indices = torch.argmax(overlap_mat, dim=1)
                        # (frist_n, num_valid_gt): (i, j) == 1 indicates the j-th
                        # gt is the best match for the i-th bbox
                        max_overlap_mask = eye_matrix[max_overlap_indices]
                        # second, filter the ones that larger than thrrshold but not 
                        # the best for bbox
                        overlap_score = overlap_score.mul(max_overlap_mask)
                        # (num_valid_gt, ): which is the best fit for every gt?
                        max_score_indices = torch.argmax(overlap_score, dim=0)
                        # (first_n, )
                        output = torch.zeros((self.first_n,))
                        # the intersection of max_score bbox and valid bbox
                        output[torch.LongTensor(np.intersect1d(
                            max_score_indices.cpu(), valid_bbox_indices.cpu()
                        ))] = 1
                        output_list_per_class.append(output[:,None, None].to(self.device))
                    # (first_n, 1, num_thresh)
                    output_list_per_image.append(torch.cat(output_list_per_class, dim=2))
            # (first_n, num_classes, num_thresh)
            output_list.append(torch.cat(output_list_per_image, dim=1)[None,...])
        # (batch_images, first_n, num_classes, num_thresh)
        output = torch.cat(output_list, dim=0)
        return output
        
    def relationnet_inference(boxes, scores, targets, image_shapes):
        """
        Args:
            boxes (Tensor): (batch_images, first_n, num_classes, 4)
            scores (Tensor): (batch_images, first_n, num_classes, num_thresh)
            targets (Tensor): (batch_images, first_n, num_classes, num_thresh)
            image_shapes (List[Tuple]): A list of (width, height) tuples for each image in the batch.
        Return:
            result (List[Instances]): 
              - pred_boxes (Boxes): (num_pred, 4)
              - scores (Tensor): (num_pred, num_classes)
              - pred_classes (Tensor): (num_pred,)
            filter_indices (Tensor)
        """
        thresh_idx = int(troch.where(self.iou_thresh == self.nms_thresh_test)[0][0])
        batch_images, first_n, num_classes = boxes.shape[:3]
        scores = scores[...,thresh_idx]
        results, filter_indices = [], []
        for batch_idx in range(batch_images):
            filter_idx = targets[batch_idx,:,:,thresh_idx].nonzero()[:, 0]
            result = Instances(image_shapes[batch_idx])
            mask_idx = filter_idx.split(1, dim=1)
            pred_boxes = Boxes(boxes[batch_idx,...][mask_idx].view(-1, 4))
            pred_boxes.clip(image_shapes[batch_idx])
            result.pred_boxes = pred_boxes
            result.scores = scores[batch_idx,...][mask_idx]
            result.pred_classes = filter_idx[:, 1]
            results.append(result)
            filter_indices.append(filter_idx)
        return results, filter_indices
    
    def nms_relation_loss(self, nms_multi_score, nms_multi_target):
        nms_pos_loss = - nms_multi_target.mul(torch.log(nms_multi_score + self.nms_eps))
        nms_neg_loss = - (1.0 - nms_multi_target).mul(torch.log(1.0 - nms_multi_score + self.nms_eps))
        normalizer = self.first_n * self.num_thresh
        nms_pos_loss = self.nms_pos_scale * nms_pos_loss / normalizer
        nms_neg_loss = nms_neg_loss / normalizer
        loss = torch.mean(nms_pos_loss + nms_neg_loss) * self.nms_loss_scale
        return loss
            
    def forward(
        self, roi_feat, predictions, proposal_boxes, num_boxes
    ):
        """
        Args:
            predictions (Tuple(Tensor)): (cls_score, bbox_pred)
              - scores (Tensor): (all_valid_boxes, num_classes + 1)
              - proposal_deltas (Tensor): (all_valid_boxes, num_reg_classes * 4)
            proposal_boxes (List(Boxes)): proposed boxes
        Return:
            nms_multi_score (Tensor): s_0*s_1 in the paper
            sorted_bbox
            sorted_score
        """

        # TODO: remove groud truth after added in query
        scores, proposal_deltas = predictions
        proposal_deltas = proposal_deltas.detach()
        all_valid_boxes = proposal_deltas.shape[0]
        batch_images = all_valid_boxes // num_boxes
        assert all_valid_boxes % num_boxes == 0
        # batch_images = len(boxes_per_image)
        proposal_deltas.require_grad = False
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        # (all_valid_boxes, 4, num_reg_classes)
        refined_bbox = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        ).view(all_valid_boxes, 4, self.num_reg_classes)
        # (batch_images, num_boxes, num_classes)
        # scores_pad = padding_tensor(scores, boxes_per_image, num_boxes)
        scores = scores.view(batch_images, num_boxes, self.num_classes + 1)
        # (batch_images, num_boxes, num_classes)
        sorted_score, rank_indices = F.softmax(
            scores[...,:-1], dim = 1
        ).sort(dim=1, descending=True)
        # (batch_images, first_n, num_classes)
        sorted_score, rank_indices = sorted_score[:,:self.first_n,:], rank_indices[:,:self.first_n,:]
        # (batch_images, first_n, num_classes)
        rank_indices = rank_indices.add(
            torch.arange(batch_images)[:,None,None].to(self.device) * num_boxes
        ).view(-1).to(torch.int64)
        # (batch_images, num_boxes, 4, num_reg_classes)
        # refined_bbox_pad = padding_tensor(refined_bbox, boxes_per_image, num_boxes)
        # (batch_images * num_boxes, 4, num_reg_classes)
        # refined_bbox = refined_bbox_pad.view(-1, 4, self.num_reg_classes)
        # (batch_images, first_n, num_classes, 4, num_reg_classes)
        sorted_bbox = refined_bbox[rank_indices].view(
            batch_images, self.first_n, self.num_classes, 4, self.num_reg_classes
        )
        # (batch_images, first_n, num_classes, 4) or original shape
        sorted_bbox = sorted_bbox.squeeze(4) # num_reg_classes may be 1
        if len(sorted_bbox.shape) == 5:
            # (num_classes, num_reg_classes, batch_images, first_n, 4)
            # => (batch_images, first_n, 4, nuum_classes)
            sorted_bbox = sorted_bbox.permute(2,4,0,1,3).diagonal(0)
            # (batch_images, first_n, num_classes, 4)
            sorted_bbox = sorted_bbox.transpose(2, 3)
        # (frist_n, 1024)
        nms_rank_embedding = extract_rank_embedding(self.first_n, 1024, self.device)
        # (frist_n, 128)
        nms_rank_feat = self.rank_emb_fc(nms_rank_embedding)
        # (batch_images, num_classes, first_n, first_n, 4)
        
        nms_position_matrix = extract_multi_position_matrix(sorted_bbox, self.device)
        # (batch_images, num_classes, first_n, first_n, fc_dim)
        nms_position_embedding = extract_pairwise_multi_position_embedding(
            nms_position_matrix,  self.pos_emb_dim, self.device
        )
        # (all_valid_boxes, feat_dim) => (all_valid_boxes, nms_fc_dim)
        roi_feat_emb = self.roi_emb_fc(roi_feat)
        # (batch_images, num_boxes, nms_fc_dim)
        # roi_feat_pad = padding_tensor(roi_feat_emb, boxes_per_image, num_boxes)
        # (batch_images * first_n * num_classes, nms_fc_dim)
        roi_feat = roi_feat_emb.view(-1, self.nms_fc_dim)[rank_indices]
        # (batch_images, first_n, num_classes, nms_fc_dim)
        roi_feat = roi_feat.view(batch_images, self.first_n, self.num_classes, self.nms_fc_dim)
        #################### vectorized nms ####################
        # (batch_images, first_n, num_classes, nms_fc_dim)
        nms_embedding_feat = nms_rank_feat[None,:,None,:] + roi_feat
        # nms_attention: (batch_images, first_n, num_classes, dim[2])
        # nms_softmax: (batch_images, num_classes * fc_dim, first_n, first_n)
        nms_attention, nms_softmax = self.attention_module_nms_multi_head(
            nms_embedding_feat, nms_position_embedding
        )
        # (batch_images, first_n, num_classes, nms_fc_dim) 
        # => (batch_images, first_n * num_classes, nms_fc_dim)
        nms_all_feat = F.relu(
            nms_embedding_feat + nms_attention
        ).view(batch_images, self.first_n * self.num_classes, self.nms_fc_dim)
        # (batch_images, first_n * num_classes, num_thresh)
        # => (batch_images, first_n, num_classes, num_thresh)
        nms_conditional_logit = self.logit_fc(
            nms_all_feat
        ).view(batch_images, self.first_n, self.num_classes, self.num_thresh)
        # (batch_images, first_n, num_classes, num_thresh)
        nms_conditional_score = F.sigmoid(nms_conditional_logit)
        # (batch_images, first_n, num_classes, num_thresh)
        nms_multi_score = nms_conditional_score + sorted_score[...,None]
        nms_conditional_score = nms_conditional_score.detach()
        return nms_multi_score, sorted_bbox, sorted_score