import torch
import numpy as np

__all__ = ["extract_position_embedding", "extract_position_matrix", "extract_rank_embedding", "extract_multi_position_matrix", "extract_pairwise_multi_position_embedding"]

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
	batch_images, num_boxes = position_mat.shape[:2]
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
	widths = (boxes[...,2]-boxes[...,0]).view(batch_images, num_boxes, 1) 
	heights = (boxes[...,3] - boxes[...,1]).view(batch_images, num_boxes, 1)
	# (batch_images, num_boxes, 1)
	x_center = 0.5 * boxes[..., 0::2].sum(dim=2).view(batch_images, num_boxes, 1)
	y_center = 0.5 * boxes[..., 1::2].sum(dim=2).view(batch_images, num_boxes, 1)
	# (batch_images, num_boxes, num_boxes)
	x_delta = x_center.sub(x_center.transpose(1, 2)).div(widths)
	y_delta = y_center.sub(y_center.transpose(1, 2)).div(heights)	
	# (batch_images, num_boxes, num_boxes)
	pos1 = torch.log(torch.max(x_delta.abs(), torch.full(x_delta.shape, eps, dtype=torch.float32).cuda())).unsqueeze(3)
	pos2 = torch.log(torch.max(y_delta.abs(), torch.full(x_delta.shape, eps, dtype=torch.float32).cuda())).unsqueeze(3)
	pos3 = torch.log(widths.div(widths.transpose(1, 2))).unsqueeze(3)
	pos4 = torch.log(heights.div(heights.transpose(1, 2))).unsqueeze(3)
	# (batch_images, num_boxes, num_boxes, 4)
	position_matrix = torch.cat([pos1, pos2, pos3, pos4], dim = 3)
	assert position_matrix.shape == (batch_images, num_boxes, num_boxes, 4)
	return position_matrix


def extract_rank_embedding(rank_dim, feat_dim, wave_length=1000):
	""" 
	Args:
		rank_dim (int): maximum of ranks
		feat_dim (int): dimension of embedding feature
		wave_length (int): wave length
	Returns:
		embedding (Tensor): (rank_dim, feat_dim)
	"""
	rank_range = torch.arange(0, rank_dim).cuda()
	feat_range = torch.arange(0, feat_dim / 2).cuda()
	dim_mat = torch.full((1,), wave_length, dtype=torch.float32).cuda().pow((2. / feat_dim) * feat_range).view(1, -1)
	rank_mat = rank_range[:,None,...].cuda()
	div_mat = rank_mat.div(dim_mat)
	sin_mat = torch.sin(div_mat)
	cos_mat = torch.cos(div_mat)
	embedding = torch.cat([sin_mat, cos_mat], dim=1)
	return embedding

def extract_pairwise_multi_position_embedding(position_mat, emb_dim, wave_length=1000):
	"""
	Extract the position embedding from the position matrix
	Args:
		position_matrix (Tensor): (batch_images, num_classes, num_boxes, num_boxes, 4)
		emb_dim (int): the length of position embedding vector
	Return:
		position_embedding (Tensor): (batch_images, num_classes, num_boxes, num_boxes, emb_dim)
	"""
	# TODO: support batch
	batch_images, num_classes, num_boxes = position_mat.shape[:3]
	assert position_mat.shape[2] == position_mat.shape[3]
	# (emb_dim/8,)
	feat_range = torch.arange(emb_dim // 8).cuda()
	# (1, 1, 1, 1, 1, emb_dim / 8)
	dim_mat = torch.Tensor([wave_length]).cuda().pow((8./emb_dim) * feat_range).view(1, 1, 1, 1, 1, -1)
	# (batch_images, num_classes, num_boxes, num_boxes, 4, 1) / (1, 1, 1, 1, 1, emb_dim/8) 
	# => (batch_images, num_classes, num_boxes, num_boxes, 4, emb_dim/8)
	div_mat = (100.0 * position_mat).unsqueeze(5).div(dim_mat)
	sin_mat = torch.sin(div_mat)
	cos_mat = torch.cos(div_mat)
	# (batch_images, num_classes, num_boxes, num_boxes, 4, emb_dim/4) 
	# => (batch_images, num_classes, num_boxes, num_boxes, emb_dim)
	embedding = torch.cat([sin_mat, cos_mat], dim=5).view(batch_images, num_classes, num_boxes, num_boxes, emb_dim)	
	return embedding


def extract_multi_position_matrix(boxes):
	""" Extract multi-class position matrix
	Args:
		boxes: (batch_images, num_boxes, num_classes, 4)
	Returns:
		position_matrix: (bath_images, num_classes, num_boxes, num_boxes, 4)
	"""
	eps = 1e-3
	batch_images, num_boxes, num_classes = boxes.shape[:3]
	# (batch_images, num_classes, num_boxes, 1)
	widths = (boxes[...,2] - boxes[...,0]).view(batch_images, num_classes, num_boxes, 1) 
	heights = (boxes[...,3] - boxes[...,1]).view(batch_images, num_classes, num_boxes, 1)
	# (batch_images, num_boxes, num_classes, 1)
	x_center = 0.5 * boxes[..., 0::2].sum(dim=3).view(batch_images, num_classes, num_boxes, 1)
	y_center = 0.5 * boxes[..., 1::2].sum(dim=3).view(batch_images, num_classes, num_boxes, 1)
	# (batch_images, num_classes, num_boxes, num_boxes)
	x_delta = x_center.sub(x_center.transpose(2, 3)).div(widths)
	y_delta = y_center.sub(y_center.transpose(2, 3)).div(heights)	
	# (batch_images, num_classes, num_boxes, num_boxes)
	pos1 = torch.log(torch.max(x_delta.abs(), torch.full(x_delta.shape, eps, dtype=torch.float32).cuda())).unsqueeze(4)
	pos2 = torch.log(torch.max(y_delta.abs(), torch.full(y_delta.shape, eps, dtype=torch.float32).cuda())).unsqueeze(4)
	pos3 = torch.log(widths.div(widths.transpose(2, 3))).unsqueeze(4)
	pos4 = torch.log(heights.div(heights.transpose(2, 3))).unsqueeze(4)
	# (batch_images, num_classes, num_boxes, num_boxes, 4)
	position_matrix = torch.cat([pos1, pos2, pos3, pos4], dim = 4)
	assert position_matrix.shape == (batch_images, num_classes, num_boxes, num_boxes, 4)
	return position_matrix
	

