import os
from detectron2.config import CfgNode as CN
from detectron2 import model_zoo

def add_relationnet_config(cfg):
	"""
	Add config for relationnet.
	"""
	_C = cfg
	_C.merge_from_file(os.path.join(os.path.abspath("./configs/relationnet_faster_R_50_C4.yaml")))
	_C.MODEL.RELATIONNET = CN()
	_C.MODEL.RELATIONNET.FEAT_DIM = 1024
	_C.MODEL.RELATIONNET.LEARN_NMS_TRAIN = False
	_C.MODEL.RELATIONNET.LEARN_NMS_TEST = False
	_C.MODEL.RELATIONNET.NUM_RELATION = 1
	_C.MODEL.RELATIONNET.POS_EMB_DIM = 64
	_C.MODEL.RELATIONNET.ATT_FC_DIM = 16
	_C.MODEL.RELATIONNET.ATT_GROUPS = 16
	_C.MODEL.RELATIONNET.ATT_DIM = (1024, 1024, 1024)
	_C.MODEL.RELATIONNET.FIRST_N_TRAIN = 100
	_C.MODEL.RELATIONNET.FRIST_N_TEST = 0
	_C.MODEL.RELATIONNET.FC_DIM = 128
	_C.MODEL.RELATIONNET.NMS_THRESH = [0.5, 0.75]
	_C.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
	_C.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)



	
