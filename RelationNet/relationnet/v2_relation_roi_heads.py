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
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Instances
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

__all__ = ["RelationROIHeads", "AttentionModule", "extract_position_embedding", "extract_position_matrix"]

def extract_position_embedding(position_mat, emb_dim, wave_length=1000):
	"""
	Extract the position embedding from the position matrix
	Args:
		position_matrix (Tensor): shape (batch_images, num_boxes, num_boxes, 4), representing geometry 
		feature of all bboxes
		emb_dim (int): the length of position embedding vector
	Return:
		position_embedding (Tensor): shape (batch_images, num_boxes, num_boxes, emb_dim), representing 
		the corresponding position embedding of all bboxes
	"""
	# TODO: support batch
	batch_images, num_boxes = position_mat.shape[0], position_mat.shape[1]
	assert position_mat.shape[1] == position_mat.shape[2]
	# (emb_dim/8,)
	feat_range = torch.arange(emb_dim // 8).cuda()
	# (batch_images, num_boxes, num_boxes, 4) / (1, 1, 1, 1, emb_dim/8) 
	# => (batch_images, num_boxes, num_boxes, 4, emb_dim/8)
	dim_mat = torch.Tensor([wave_length]).cuda().pow((8./emb_dim) * feat_range).view(1, 1, 1, 1, -1)
	# (batch_images, num_boxes, num_boxes, 4, emb_dim/8)
	div_mat = (100.0 * position_mat).unsqueeze(4).div(dim_mat)
	sin_mat = torch.sin(div_mat)
	cos_mat = torch.cos(div_mat)
	# (batch_images, num_boxes, num_boxes, 4, emb_dim/4) => (batch_images, num_boxes, num_boxes, emb_dim)
	embedding = torch.cat([sin_mat, cos_mat], dim=4).view(batch_images, num_boxes, num_boxes, emb_dim)
	return embedding

def extract_position_matrix(proposal_boxes):
	"""
	Extract the position matrix from proposal boxes in batch images
	Args:
		proposal_boxes (List[Boxes]): proposal boxes in batch images
	Return:
		position_matrix (Tensor): (batch_images, num_boxes, num_boxes, 4)
	"""
	# TODO: support batch
	eps, num_boxes, batch_images = 1e-3, proposal_boxes[0].tensor.shape[0], len(proposal_boxes)
	# (batch_images, num_boxes, 4)
	boxes = torch.cat([boxes.tensor[None,...] for boxes in proposal_boxes])
	# (batch_images, num_boxes, 1), adding 1. to avoid deviding by zero
	widths = (boxes[...,2]-boxes[...,0] + 1.).view(batch_images, num_boxes, 1) 
	heights = (boxes[...,3] - boxes[...,1] + 1.).view(batch_images, num_boxes, 1)
	# (batch_images, num_boxes, 1)
	x_center = 0.5 * boxes[..., 0::2].sum(dim=2).view(batch_images, num_boxes, 1)
	y_center = 0.5 * boxes[..., 1::2].sum(dim=2).view(batch_images, num_boxes, 1)
	# (batch_images, num_boxes, num_boxes)
	x_delta = x_center.sub(x_center.transpose(1, 2)).div(widths)
	y_delta = y_center.sub(y_center.transpose(1, 2)).div(heights)
	# (batch_images, num_boxes, num_boxes)
	pos1 = torch.log(torch.max(x_delta.abs(), torch.Tensor([[[eps]]]).cuda())).unsqueeze(3)
	pos2 = torch.log(torch.max(y_delta.abs(), torch.Tensor([[[eps]]]).cuda())).unsqueeze(3)
	pos3 = torch.log(widths.div(widths.transpose(1, 2))).unsqueeze(3)
	pos4 = torch.log(heights.div(heights.transpose(1, 2))).unsqueeze(3)
	# (batch_images, num_boxes, num_boxes, 4)
	position_matrix = torch.cat([pos1, pos2, pos3, pos4], dim = 3)
	assert position_matrix.shape == (batch_images, num_boxes, num_boxes, 4)
	return position_matrix

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
		# parameters
		self.fc_dim, self.emb_dim = fc_dim, emb_dim
		self.feat_dim, self.dim, self.groups =	feat_dim, dim, groups
		assert self.fc_dim == self.groups, "The fc_dim doesn't match groups!"
		assert self.dim[0] == self.dim[1], "Matrix multiply requires same dimensions!"
		assert dim[0] % groups == 0 and dim[2] % groups == 0
		self.dim_groups = (dim[0]//groups, dim[1]//groups, dim[2]//groups)	
		self.num_reg_classes = num_reg_classes
		# modules
		self.pos_fc = nn.Linear(emb_dim, fc_dim).to(device)
		self.query_fc = nn.Linear(feat_dim, dim[0]).to(device)
		self.key_fc = nn.Linear(feat_dim, dim[1]).to(device)
		self.conv1x1_out = nn.Conv2d(fc_dim * feat_dim, dim[2], (1,1), groups=fc_dim).to(device)
		# initialization
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
		aff_weight  = F.relu(self.pos_fc(position_embedding)).view(
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
			aff_weight, torch.Tensor([[[1e-6]]]).cuda()
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

@ROI_HEADS_REGISTRY.register()
class RelationROIHeads(Res5ROIHeads):
	def __init__(self, cfg, input_shape):
		"""
		Args:
			num_ralation (int): the number of relation modules used. Each with 
			seperate parameters
		"""
		super().__init__(cfg, input_shape)
		# parameters
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
		self.num_boxes = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
		# modules
		self.res5, self.res5_out_channels = self._build_res5_block(cfg)
		self.res5_out_size = self.res5_out_channels * self.pooler_resolution ** 2 // 4 
		self.box_predictor = FastRCNNOutputLayers(
			cfg, ShapeSpec(channels=self.feat_dim, height=1, width=1)
		)
		self.attention_module_multi_head = {
			0: [self._build_attention_module_multi_head() for i in range(self.num_relation)],
			1: [self._build_attention_module_multi_head() for i in range(self.num_relation)]
		}
		# self.attention_module_nms_multi_head = {
		#	  1: self._build_attention_nms_module() for i in range(self.num_relation),
		#	  2: self._build_attention_nms_module() for i in range(self.num_relation)
		# }
		self.fc_feat = nn.Linear(
			self.res5_out_size,
			self.feat_dim
		).to(device)
		self.fc = [nn.Linear(self.feat_dim, self.feat_dim).to(device) for i in range(2)]
		self.cls_fc = nn.Linear(self.feat_dim, self.num_classes + 1).to(device)  
		self.pred_fc = nn.Linear(self.feat_dim, self.num_reg_classes * 4).to(device)
		# intialization
		mean, std = 0.0, 0.01
		nn.init.normal_(self.fc_feat.weight, mean, std)
		nn.init.constant_(self.fc_feat.bias, mean)
		for i in range(2):
			nn.init.normal_(self.fc[i].weight, mean, std)
			nn.init.constant_(self.fc[i].bias, mean)
		nn.init.normal_(self.cls_fc.weight, mean, std)
		nn.init.constant_(self.cls_fc.bias, mean)
		nn.init.normal_(self.pred_fc.weight, mean, std)
		nn.init.constant_(self.pred_fc.bias, mean)
	def _build_attention_module_multi_head(self):
		attention_module_multi_head = AttentionModule(
			self.att_fc_dim, self.pos_emb_dim, self.feat_dim, self.att_dim, 
			self.att_groups, self.num_reg_classes, self.device
		)
		return attention_module_multi_head

	def _build_attention_nms_module_multi_head(self):
		pass

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
		if self.training:
			assert targets
			proposals = self.label_and_sample_proposals(proposals, targets)
			# now proposals have fields "gt_boxes" and "gt_classes"
		del targets
		# proposal_boxes: List[Boxes]
		proposal_boxes = [x.proposal_boxes for x in proposals]
		proposal_boxes_pad = [
			Boxes(F.pad(
				x.tensor, (self.num_boxes - x.tensor.shape[0], 0)
			)) for x in proposal_boxes
		]
		# mask[num_boxes * i + j] == True means the j-th box in i-th image is valid
		mask = (torch.cat([boxes.tensor for boxes in proposal_boxes_pad]).mean(dim=1) != 0)
		# (batch_images, num_boxes, channels * outshape1 * outshape2)
		box_features = self._shared_roi_transform(
			[features[f] for f in self.in_features], proposal_boxes
		).view(len(proposal_boxes), self.num_boxes, -1)

		################################ 2fc+RM Head ################################
		# Input:
		#	  box_features (Tensor): (batch_images*num_boxes, channels, outshape1, outshape2)
		#	  proposal_boxes (List[Boxes]): has batch_images instances
		# Output:
		#	  rois:
		#	  cls_prob:
		#	  bbox_pred:

		# (batch_images, num_boxes, feat_dim)
		fc_out = self.fc_feat(box_features)
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
		#	  - cls_score (Tensor): (all_boxes, num_classes + 1)
		#	  - bbox_pred (Tensor): (all_boxes, num_reg_classes*4)
		predictions = self.box_predictor(fc_out)

		######################### learn nms #########################
		# 
		# Input is a set of detected objects:
		#	  Each object has its final 1024-d feature, classification score s0 and bounding box.
		# 
		# The network has three steps. 
		# 1. The 1024-d feature and classification score is fused to generate the appearance feature. 
		# 2. A relation module transforms such appearance features of all objects. 
		# 3. The transformed features of each object pass a linear classifier and sigmoid to output 
		#	 the probabilit y ∈ [0, 1].

		if self.training:
			del features
			losses = self.box_predictor.losses(predictions, proposals)

			return [], losses
		else:
			pred_instances, _ = self.box_predictor.inference(predictions, proposals)
			pred_instances = self.forward_with_given_boxes(features, pred_instances)
			return pred_instances, {} 
