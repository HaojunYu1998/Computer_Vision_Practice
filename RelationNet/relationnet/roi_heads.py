################################# relatioinnet/realtion_roi_heads.py #################################
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals

from .attention_module import AttentionModule
from .embedding import extract_position_embedding, extract_position_matrix
from .learn_nms import LearnNMSModule

__all__ = ["RelationROIHeads"]

@ROI_HEADS_REGISTRY.register()
class RelationROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        """
        Args:
            num_ralation (int): the number of relation modules used. Each with 
            seperate parameters
        """
        super().__init__(cfg, input_shape)
        ############################### parameters #################################
        if self.training:
            self.pre_nms_dim = cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN
        else:
            self.pre_nms_dim = cfg.MODEL.RPN.PRE_NMS_TOPK_TEST
        self.num_relation = cfg.MODEL.RELATIONNET.NUM_RELATION
        self.pos_emb_dim = cfg.MODEL.RELATIONNET.POS_EMB_DIM
        self.feat_dim = cfg.MODEL.RELATIONNET.FEAT_DIM
        self.att_fc_dim = cfg.MODEL.RELATIONNET.ATT_FC_DIM
        self.att_groups = cfg.MODEL.RELATIONNET.ATT_GROUPS
        self.att_dim = cfg.MODEL.RELATIONNET.ATT_DIM
        self.pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.num_reg_classes = self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG:
            self.num_reg_classes = 2
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.learn_nms_train = cfg.MODEL.RELATIONNET.LEARN_NMS_TRAIN
        self.learn_nms_test = cfg.MODEL.RELATIONNET.LEARN_NMS_TEST
        self.first_n = cfg.MODEL.RELATIONNET.FIRST_N_TEST
        self.num_boxes = self.batch_size_per_image
        if self.training:
            self.first_n = cfg.MODEL.RELATIONNET.FIRST_N_TRAIN
        ############################### modules ####################################
        self.res5, self.res5_out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=self.res5_out_channels, height=1, width=1)
        )
        self.fc_feat = nn.Linear(self.res5_out_channels, self.feat_dim).to(self.device)
        self.fc = [nn.Linear(self.feat_dim, self.feat_dim).to(self.device) for i in range(2)]
        self.nms_module = LearnNMSModule(cfg)
        ########################## freeze parameters ###############################
        # for block in self.res5:
        #     block.freeze()
        # for p in self.box_predictor.parameters():
        #     p.requires_grad = False
        ############################# intialization ################################
        mean, std = 0.0, 0.01
        nn.init.normal_(self.fc_feat.weight, mean, std)
        nn.init.constant_(self.fc_feat.bias, mean)
        for i in range(2):
            nn.init.normal_(self.fc[i].weight, mean, std)
            nn.init.constant_(self.fc[i].bias, mean)

    def _build_attention_module_multi_head(self):
        attention_module_multi_head = AttentionModule(
            self.att_fc_dim, self.pos_emb_dim, self.feat_dim, self.att_dim, 
            self.att_groups, self.num_reg_classes, self.device
        )
        return attention_module_multi_head
    
    @torch.no_grad()
    def label_proposals(self, proposals, targets):
        proposals_with_gt = []
        self.num_boxes = np.min([len(x.proposal_boxes) for x in proposals])
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            _, indices = torch.sort(proposals_per_image.objectness_logits, descending=True)
            sampled_idxs = indices[:self.num_boxes]
            proposals_per_image = proposals_per_image[sampled_idxs]
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            gt_classes = self._label_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[matched_idxs])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            proposals_with_gt.append(proposals_per_image)
        return proposals_with_gt
            
    def _label_proposals(self, matched_idxs, matched_labels, gt_classes):
        has_gt = gt_classes.numel() > 0
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels <= 0] = self.num_classes
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
        return gt_classes
    
    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList)
            features (dict[str,Tensor]):
                key: str like ["p2", "p3", "p4", "p5"] or ["res4"]
                value: Tensor.shape = (N, C, H, W)
            proposals (list[Instances]):
                Each Instances contains bboxes/masks/keypoints of a image. We focus on
                    - proposal_boxes: proposed bboxes in format `Boxes`
                    - objectness_logits: list[np.ndarray] each is an N sized array of 
                      objectness scores corresponding to the boxes

            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            pred_instances (list[Instances]): length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            loss (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        # TODO: index the nms_multi_target to get the corresponding "first_n"
        # complete the binary cross_entropy loss
        del images
        if self.training:
            assert targets
            proposals = self.label_proposals(proposals, targets)
        # proposal_boxes: List[Boxes]
        proposal_boxes = [x.proposal_boxes for x in proposals]
        # (all_valid_boxes, channels, outshape1, outshape2)
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        # (all_valid_boxes, channels * outshape1 * outshape2)
        # box_features = box_features.view(box_features.shape[0], -1)
        
        ################################ 2fc+RM Head ###############################
        # Input:
        #     box_features (Tensor): (batch_images*num_boxes, channels, outshape1, outshape2)
        #     proposal_boxes (List[Boxes]): has batch_images instances
        # Output:
        #     rois:
        #     cls_prob:
        #     bbox_pred:
        # TODO: add ground truth boxes in query
        
        fc_out = self.fc_feat(box_features.mean(dim=[2, 3]))
       
        ############################### learn nms ##################################
        # 
        # Input is a set of detected objects:
        #     Each object has its final 1024-d feature, classification score s0 and bounding boxes.
        # 
        # The network has three steps. 
        # 1. The 1024-d feature and classification score is fused to generate the appearance feature. 
        # 2. A relation module transforms such appearance features of all objects. 
        # 3. The transformed features of each object pass a linear classifier and sigmoid to output 
        #    the probabilit y ∈ [0, 1].
        
        # predictions: (cls_score, bbox_pred)
        #   - scores (Tensor): (all_valid_boxes, num_classes + 1), [0, num_classes] 
        #     => num_classes indicates backgroud
        #   - proposal_deltas (Tensor): (all_valid_boxes, num_reg_classes * 4)
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        # do not use learn_nms
        if self.training and (not self.learn_nms_train):
            raise NoImplementationError("training should set learn_nms == True!")
        elif (not self.training) and (not self.learn_nms_test):
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
        
        # nms_multi_score: (batch_images, first_n, num_classes, num_thresh)
        # sorted_boxes: (batch_images, first_n, num_classes, 4)
        # sorted_score: (batch_images, first_n, num_classes)
        nms_multi_score, sorted_boxes, sorted_score = self.nms_module(
            fc_out, predictions, proposal_boxes, self.num_boxes
        )
        # (batch_images, first_n, num_classes, num_thresh)
        nms_multi_target = self.nms_module.get_multi_target(sorted_boxes, targets, sorted_score)
        nms_multi_target = nms_multi_target.detach()
        del targets
        ############################# construct losses ################################
        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            losses["loss_relation"] = self.nms_module.nms_relation_loss(nms_multi_score, nms_multi_target)
            return [], losses
        else:
            pred_instances = self.nms_module.relationnet_inference(
                sorted_boxes, nms_multi_score, nms_multi_target,
                image_shapes = [x.image_size for x in proposals],
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
