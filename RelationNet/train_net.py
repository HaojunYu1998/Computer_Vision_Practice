import os
import torch
import torch.nn.functional as F
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import PascalVOCDetectionEvaluator

from relationnet import add_relationnet_config

# class Trainer(DefaultTrainer):
# 	@classmethod
# 	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
# 		if output_folder is None:
# 			output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
# 		return PascalVOCDetectionEvaluator(dataset_name)   

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return PascalVOCDetectionEvaluator(dataset_name)

def setup(args):
	"""
	Create configs and perform basic setups.
	"""
	cfg = get_cfg()
	cfg.merge_from_file(
		model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")
	)
	cfg.merge_from_list(args.opts)
	add_relationnet_config(cfg)
	cfg.freeze()
	default_setup(cfg, args)
	return cfg


def main(args):
	cfg = setup(args)

	if args.eval_only:
		model = Trainer.build_model(cfg)
		DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
			cfg.MODEL.WEIGHTS, resume=args.resume
		)
		res = Trainer.test(cfg, model)
		return res

	trainer = DefaultTrainer(cfg)
	trainer.resume_or_load(resume=args.resume)
	return trainer.train()


if __name__ == "__main__":
	args = default_argument_parser().parse_args()
	print("Command Line Args:", args)
	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)
  
