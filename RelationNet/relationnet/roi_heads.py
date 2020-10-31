################################# relatioinnet/realtion_roi_heads.py #################################
import inspect
import logging
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
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.structures import Instances
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from .attention_module import AttentionModule, AttentionNMSModule
from .embedding import (
	extract_position_embedding,
	extract_position_matrix, 
	extract_rank_embedding, 
	extract_multi_position_matrix,
	extract_pairwise_multi_position_embedding,
)
from .padding import padding_tensor

__all__ = ["RelationROIHeads"]

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads_v2(Res5ROIHeads):
	def forward(self, images, features, proposals, targets=None):
		"""
		Args:
			images (ImageList):
			features (dict[str,Tensor]): input data as a mapping from feature
				map name to tensor. Axis 0 represents the number of images `N` in
				the input data; axes 1-3 are channels, height, and width, which may
				vary between feature maps (e.g., if a feature pyramid is used).
			proposals (list[Instances]): length `N` list of `Instances`. The i-th
				`Instances` contains object proposals for the i-th input image,
				with fields "proposal_boxes" and "objectness_logits".
			targets (list[Instances], optional): length `N` list of `Instances`. The i-th
				`Instances` contains the ground-truth per-instance annotations
				for the i-th input image.  Specify `targets` during training only.
				It may have the following fields:

				- gt_boxes: the bounding box of each instance.
				- gt_classes: the label for each instance with a category ranging in [0, #class].
				- gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
				- gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

		Returns:
			list[Instances]: lt
			dict[str->Tensor]:
			mapping from a named loss to a tensor storing the loss. Used during training only.
		"""
		del images

		if self.training:
			assert targets
			print(targets[0].gt_boxes.tensor.shape)
			proposals = self.label_and_sample_proposals(proposals, targets)
		del targets

		proposal_boxes = [x.proposal_boxes for x in proposals]
		# box_features is a torch.Tensor with shape (M, C, outshap0, outshape1)
		# where M is the total number of boxes aggregated over all N batch images.
		box_features = self._shared_roi_transform(
			[features[f] for f in self.in_features], proposal_boxes
		)
		predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

		if self.training:
			del features
			losses = self.box_predictor.losses(predictions, proposals)
			if self.mask_on:
				proposals, fg_selection_masks = select_foreground_proposals(
					proposals, self.num_classes
				)
				# Since the ROI feature transform is shared between boxes and masks,
				# we don't need to recompute features. The mask loss is only defined
				# on foreground proposals, so we need to select out the foreground
				# features.
				mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
				del box_features
				losses.update(self.mask_head(mask_features, proposals))
			return [], losses
		else:
			pred_instances, _ = self.box_predictor.inference(predictions, proposals)
			pred_instances = self.forward_with_given_boxes(features, pred_instances)
			return pred_instances, {}


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
		self.num_reg_classes = 2 if cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG else self.num_classes
		self.device = device = torch.device(cfg.MODEL.DEVICE)
		self.learn_nms_train = cfg.MODEL.RELATIONNET.LEARN_NMS_TRAIN
		self.learn_nms_test = cfg.MODEL.RELATIONNET.LEARN_NMS_TEST
		self.nms_thresh = cfg.MODEL.RELATIONNET.NMS_THRESH
		self.num_thresh = len(self.nms_thresh)
		self.first_n = cfg.MODEL.RELATIONNET.FIRST_N_TRAIN if self.training else cfg.MODEL.RELATIONNET.FIRST_N_TEST
		self.bbox_means = cfg.BBOX_MEANS if self.training else None
		self.bbox_stds = cfg.BBOX_STDS if self.training else None
		self.nms_fc_dim = cfg.MODEL.RELATIONNET.FC_DIM
		################################## modules ####################################
		self.res5, self.res5_out_channels = self._build_res5_block(cfg)
		self.res5_out_size = self.res5_out_channels * self.pooler_resolution **2 // 4
		self.box_predictor = FastRCNNOutputLayers(
			cfg, ShapeSpec(channels=self.feat_dim, height=1, width=1)
		)
		self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
		self.attention_module_multi_head = {
			0: [self._build_attention_module_multi_head() for i in range(self.num_relation)],
			1: [self._build_attention_module_multi_head() for i in range(self.num_relation)]
		}
		self.attention_module_nms_multi_head = self._build_attention_nms_module_multi_head()
		self.fc_feat = nn.Linear(
			self.res5_out_size,
			self.feat_dim
		).to(device)
		self.fc = [nn.Linear(self.feat_dim, self.feat_dim).to(device) for i in range(2)]
		self.rank_emb_fc = nn.Linear(self.feat_dim, self.nms_fc_dim).to(device)
		self.roi_emb_fc = nn.Linear(self.feat_dim, self.nms_fc_dim)
		self.nms_logit_fc = nn.Linear(self.att_dim[2], self.num_thresh)
		################################ intialization ###################################
		mean, std = 0.0, 0.01
		nn.init.normal_(self.fc_feat.weight, mean, std)
		nn.init.constant_(self.fc_feat.bias, mean)
		for i in range(2):
			nn.init.normal_(self.fc[i].weight, mean, std)
			nn.init.constant_(self.fc[i].bias, mean)
		nn.init.normal_(self.rank_emb_fc.weight, mean, std)
		nn.init.constant_(self.rank_emb_fc.bias, mean)
		nn.init.normal_(self.roi_emb_fc.weight, mean, std)
		nn.init.constant_(self.roi_emb_fc.bias, mean)

	def _build_attention_module_multi_head(self):
		attention_module_multi_head = AttentionModule(
			self.att_fc_dim, self.pos_emb_dim, self.feat_dim, self.att_dim, 
			self.att_groups, self.num_reg_classes, self.device
		)
		return attention_module_multi_head

	def _build_attention_nms_module_multi_head(self):
		attention_module_nms_multi_head = AttentionNMSModule(
			fc_dim=self.att_fc_dim, 
			feat_dim=self.nms_fc_dim, emb_dim=self.pos_emb_dim,
			dim=(self.att_dim[0], self.att_dim[1], self.nms_fc_dim), 
			groups=self.att_groups, num_reg_classes=self.num_reg_classes, 
			device=self.device
		)
		return attention_module_nms_multi_head
	
	def learn_nms(self, predictions, proposal_boxes):
		"""
		Args:
			predictions (Tuple(Tensor)): (cls_score, bbox_pred)
			  - cls_score (Tensor): (all_valid_boxes, num_classes + 1)
			  - bbox_pred (Tensor): (all_valid_boxes, num_reg_classes * 4)
			proposal_boxes (List(Boxes)): proposed boxes
		Return:
			nms_multi_score (Tensor): s_0*s_1 in the paper
			sorted_bbox
			sorted_score
		"""

		# TODO: remove groud truth after added in query
		scores, proposal_deltas = predictions
		proposal_deltas.require_grad = False
		proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
		# (all_valid_boxes, 4, num_reg_classes)
		refined_bbox = self.box2box_transform.apply_deltas(
			proposal_deltas, proposal_boxes
		)
		# (batch_images, num_boxes, num_classes)
		scores_pad = padding_tensor(scores, self.nongt_per_image, padding_max_len)
		# (batch_images, first_n, num_classes)
		sorted_score, rank_indices = F.softmax(
			scores_pad[...,1:], dim=1
		).sort(dim=1, descending=True)[:,:self.first_n,:]
		# (bath_images, num_boxes, 4, num_reg_classes)
		refined_bbox_pad = padding_tensor(refined_bbox, self.nongt_per_image, padding_max_len)
		# (batch_images, first_n, num_classes, 4, num_reg_classes)
		sorted_bbox = refined_bbox_pad.take(rank_indices)
		sorted_bbox = sorted_bbox.squeeze() # num_reg_classes may be 1
		if len(sorted_bbox.shape) == 5:
			# (batch_images, first_n, num_classes, 4)
			cls_mask = torch.arange(0, num_classes).view(1,1,-1,1).repeat(batch_images, self.first_n, 1, 4)
			# (batch_images, first_n, num_classes, 4)
			sorted_bbox = sorted_bbox.gather(4, cls_mask)
		# (frist_n, 1024)
		nms_rank_embedding = extract_rank_embedding(first_n, 1024)
		# (frist_n, 128)
		nms_rank_feat = self.rank_emb_fc(nms_rank_embedding)
		# (batch_images, num_classes, first_n, first_n, 4)
		nms_position_matrix = extract_multi_position_matrix(sorted_bbox)
		# (batch_images, num_classes, first_n, first_n, fc_dim)
		nms_position_embedding = extract_pairwise_multi_position_embedding(
			nms_position_matrix,  self.pos_emb_dim
		)
		# (batch_images, first_n, num_classes, 128)
		roi_feat_emb = self.roi_emb_fc(fc_out).take(indices=rank_indices)
		#################### vectorized nms ####################
		# (batch_images, first_n, num_classes, 128)
		nms_embedding_feat = nms_rank_feat[None,:,None,:] + roi_feat_emb
		# nms_attention: (batch_images, first_n, num_classes, dim[2])
		# nms_softmax: (batch_images, num_classes * fc_dim, first_n, first_n)
		nms_attention, nms_softmax = self.attention_module_nms_multi_head(
			nms_embedding_feat, nms_position_embedding
		)
		# (batch_images, first_n, num_classes, dim[2])
		# => (batch_images, first_n * num_classes, dim[2])
		nms_all_feat = F.relu(
			nms_embedding_feat + nms_attention
		).view(batch_images, first_n * num_classes, dim[2])
		# (batch_images, first_n * num_classes, 1)
		# => (batch_images, first_n, num_classes, 1)
		nms_conditional_logit = self.nms_logit_fc(
			nms_all_feat
		).view(batch_images, first_n, num_classes, 1)
		# (batch_images, first_n, num_classes, 1)
		nms_conditional_score = F.sigmoid(nms_conditional_logit)
		# (batch_images, first_n, num_classes, 1)
		nms_multi_score = nms_conditional_score + sorted_score[...,None]
		return nms_multi_score, sorted_bbox, sorted_score

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
		del images
		# proposal_boxes: List[Boxes]
		proposal_boxes = [x.proposal_boxes for x in proposals]
		self.nongt_per_image = [x.proposal_boxes.tensor.shape[0] for x in proposals]
		# num_boxes is the max number of proposal boxes in this batch
		self.nongt_dim = self.batch_size_per_image = np.max(self.nongt_per_image)
		if self.training and self.proposal_append_gt:
			assert targets
			proposals = self.label_and_sample_proposals(proposals, targets)
			print(proposals[0].proposal_boxes.tensor.shape, proposals[1].proposal_boxes.tensor.shape)
		del targets
		proposal_boxes_pad = [
			Boxes(F.pad(
				x.tensor, (0, 0, 0, self.nongt_dim- x.tensor.shape[0])
			)) for x in proposal_boxes
		]
		# mask[num_boxes * i + j] == True means the j-th box in i-th image is valid
		mask = (torch.cat([boxes.tensor for boxes in proposal_boxes_pad]).mean(dim=1) != 0)
		# (all_valid_boxes, channels, outshape1, outshape2)
		box_features = self._shared_roi_transform(
			[features[f] for f in self.in_features], proposal_boxes
		)
		# (all_valid_boxes, channels * outshape1 * outshape2)
		box_features = box_features.view(box_features.shape[0], -1)
		batch_images, box_features_list, idx = len(proposal_boxes), [], 0
		for boxes in proposal_boxes:
			n = boxes.tensor.shape[0]
			box_features_list.append(
				F.pad(box_features[idx:idx+n,:], (0,0,0,self.num_boxes-n))
			)
		# (batch_images, num_boxes, channels * outshape1 * outshape2)
		box_features_pad = torch.cat(box_features_list).view(batch_images, self.num_boxes, -1)

		################################ 2fc+RM Head ################################
		# Input:
		#	  box_features (Tensor): (batch_images*num_boxes, channels, outshape1, outshape2)
		#	  proposal_boxes (List[Boxes]): has batch_images instances
		# Output:
		#	  rois:
		#	  cls_prob:
		#	  bbox_pred:
		# TODO: add ground truth boxes in query

		# (batch_images, num_boxes, feat_dim)
		fc_out = self.fc_feat(box_features_pad)
		# (batch_images, num_boxes, 4)
		position_matrix = extract_position_matrix(proposal_boxes_pad)
		assert position_matrix.shape[1] % self.num_boxes == 0
		# (batch_images, num_boxes, num_boxes, emb_dim)
		position_embedding = extract_position_embedding(position_matrix, self.pos_emb_dim)
		# 2fc layers
		for fc_idx in range(2):
			# (batch_images, num_boxes, feat_dim)
			fc_out = self.fc[fc_idx](fc_out)
			# (batch_images, num_boxes, feat_dim)
			attention_out = torch.zeros(fc_out.shape).to(self.device)
			# loop for realtion modules
			for att_idx in range(self.num_relation):
				attention_out += self.attention_module_multi_head[fc_idx][att_idx](
					fc_out, position_embedding
				)
			fc_out = F.relu(fc_out + attention_out)
		# (batch_images, num_boxes, feat_dim) => (all_valid_boxes, feat_dim)
		fc_out = fc_out.view(-1, self.feat_dim)[mask,...]
		# predictions: (cls_score, bbox_pred)
		#	  - cls_score (Tensor): (all_valid_boxes, num_classes + 1)
		#	  - bbox_pred (Tensor): (all_valid_boxes, num_reg_classes * 4)
		predictions = self.box_predictor(fc_out)

		# do not learn nms
		if self.training and (not self.learn_nms_train):
			del features
			losses = self.box_predictor.losses(predictions, proposals)
			return [], losses
		elif (not self.training) and (not self.learn_nms_test):
			pred_instances, _ = self.box_predictor.inference(predictions, proposals)
			pred_instances = self.forward_with_given_boxes(features, pred_instances)
			return pred_instances, {} 
		
		######################### learn nms #########################
		# 
		# Input is a set of detected objects:
		#	  Each object has its final 1024-d feature, classification score s0 and bounding boxes.
		# 
		# The network has three steps. 
		# 1. The 1024-d feature and classification score is fused to generate the appearance feature. 
		# 2. A relation module transforms such appearance features of all objects. 
		# 3. The transformed features of each object pass a linear classifier and sigmoid to output 
		#	 the probabilit y ∈ [0, 1].
		nms_eps = 1e-8
		# nms_multi_score: (batch_images, first_n, num_classes, 1)
		# sorted_boxes: (batch_images, first_n, num_classes, 4)
		# sorted_score: (batch_images, first_n, num_classes)
		nms_multi_score, sorted_bbox, sorted_score = self.learn_nms(predictions, proposal_boxes)
		nms_pos_loss = - torch.mul(nms_multi_target, torch.log(nms_multi_score + nms_eps))
		nms_neg_loss = - torch.mul((1.0 - nms_multi_target), torch.log(1.0 - nms_multi_score + nms_eps))
		normalizer = first_n 
		nms_pos_loss = cfg.TRAIN.nms_loss_scale * nms_pos_loss / normalizer
		nms_neg_loss = cfg.TRAIN.nms_loss_scale * nms_neg_loss / normalizer
		






