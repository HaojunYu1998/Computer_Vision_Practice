import os
from detectron2.config import CfgNode as CN
from detectron2 import model_zoo

def add_relationnet_config(cfg):
    """
    Add config for relationnet.
    """
    _C = cfg
    _C.DATASETS.TRAIN = ("voc_2007_trainval",)
    _C.MODEL.WEIGHTS = "/data171_hdd0/yuhj/Computer_Vision_Practice/RelationNet/output/faster_rcnn/model_final.pth"
    _C.merge_from_file(os.path.join(os.path.abspath("./configs/relationnet.yaml")))
    _C.MODEL.RELATIONNET = CN()
    _C.MODEL.RELATIONNET.FEAT_DIM = 1024
    _C.MODEL.RELATIONNET.LEARN_NMS_TRAIN = True
    _C.MODEL.RELATIONNET.LEARN_NMS_TEST = False
    _C.MODEL.RELATIONNET.NUM_RELATION = 1
    _C.MODEL.RELATIONNET.POS_EMB_DIM = 64
    _C.MODEL.RELATIONNET.ATT_FC_DIM = 16
    _C.MODEL.RELATIONNET.ATT_GROUPS = 16
    _C.MODEL.RELATIONNET.ATT_DIM = (1024, 1024, 1024)
    _C.MODEL.RELATIONNET.FIRST_N_TRAIN = 100
    _C.MODEL.RELATIONNET.FIRST_N_TEST = 200
    _C.MODEL.RELATIONNET.NMS_FC_DIM = 128
    _C.MODEL.RELATIONNET.NMS_POS_SCALE = 4.0
    _C.MODEL.RELATIONNET.NMS_LOSS_SCALE = 800.0
    _C.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
    _C.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

