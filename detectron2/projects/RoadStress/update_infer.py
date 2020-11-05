import argparse
import glob
import os
import cv2
import torch
import time
import logging
import os
import json
import copy
import numpy as np
import random

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.modeling import build_model
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from utils import *

logger = logging.getLogger("detectron2")

def processLog(path):
	totalLoss = ""
	trainingTime = ""
	modelName = ""

	logFile = open(os.path.join(path, "log.txt"))
	lines = logFile.readlines()
	for line in reversed(lines):
		if "eta" in line:
			result = line.split()
			totalLoss = result[result.index("total_loss:") + 1]
			break
	
	for line in lines:
		if "eta" in line:
			result = line.split()
			trainingTime = result[result.index("eta:") + 1]
		if "WEIGHTS: https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/" in line:
			result = line.split("/")
			modelName = result[result.index("COCO-InstanceSegmentation") + 1]
		if modelName and trainingTime and totalLoss:
			return modelName, trainingTime, totalLoss

def config(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	# Set score_threshold for builtin models
	cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
	cfg.TEST.DETECTIONS_PER_IMAGE = 500
	cfg.MODEL.WEIGHTS = args.weight
	cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000  # originally 1000
	cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 5000  # originally 1000
	#cfg.freeze()
	return cfg

def get_parser():
	parser = argparse.ArgumentParser(description="Detectron2 inference for road stress")
	parser.add_argument("--config-file", required=True, metavar="FILE", help="path to config file")
	parser.add_argument("--dataset", required=True, help="path to dataset folder")
	parser.add_argument("--weight", required=True, metavar="FILE", help="path to weight file")
	parser.add_argument("--output", help="A file or directory to save output visualizations. If not given, will show output in an OpenCV window.")
	parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score for instance predictions to be shown")
	parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER,)
	return parser

def run_on_image(predictor, image):
	predictions = predictor(img)
	vis = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get("__unused"), instance_mode=ColorMode.IMAGE)
	if "instances" in predictions:
		instances = predictions["instances"].to(torch.device("cpu"))
		vis_output = vis.draw_instance_predictions(predictions=instances)
		return predictions, vis_output
	else:
		print("Something wrong. Please check the inference.py script")

if __name__ == "__main__":
	args = get_parser().parse_args()
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	# Register the dataset:
	for d in ["old", "new"]:
		DatasetCatalog.register("roadstress_%s_val" % d, lambda d=d: get_roadstress_dicts("dataset/roadstress_%s/" % d + "/val"))
		MetadataCatalog.get("roadstress_%s_val" % d).set(thing_classes=["roadstress"])
		MetadataCatalog.get("roadstress_%s_val" % d).set(evaluator_type="coco")
		roadstress_metadata = MetadataCatalog.get("roadstress_new_val")
	print("Done Registering the dataset")

	cfg = config(args)
	cfg.DATASETS.TEST = ("roadstress_new_val", "roadstress_old_val")
	modelName, trainingTime, totalLoss = processLog(cfg.OUTPUT_DIR)
	
	print("==================================================================================")
	print("Training time: " + trainingTime)
	print("Max iterations: " + str(cfg.SOLVER.MAX_ITER))
	print("Model: " + modelName)
	print("Batch_size_per_img: " + str(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE))
	print("Anchor generator sizes: " + str(cfg.MODEL.ANCHOR_GENERATOR.SIZES))
	print("Anchor generator sizes: " + str(cfg.MODEL.ANCHOR_GENERATOR.ANGLES))
	print("Base LR: " + str(cfg.SOLVER.BASE_LR))
	print("Warmup Iter: " + str(cfg.SOLVER.WARMUP_ITERS))
	print("IMS_PER_BATCH: " + str(cfg.SOLVER.IMS_PER_BATCH))
	print("Total Loss: " + totalLoss)
	print("Finish Setting Up Config")
	
	for i in np.arange(0.05, 0.95, 0.05):
		i = i.round(decimals=2)
		print("\n\n-------------------------------------------------------------------")
		print("Threshold: " + str(i))
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(i)
		
		# create image folder
		os.makedirs(os.path.join(args.output, "threshold_" + str(i)), exist_ok=True)	

		# Inferencing
		predictor = DefaultPredictor(cfg)
		total_time = 0
		total_imgs = 0
		for d in ["old", "new"]:
			for img_path in glob.iglob(args.dataset + "roadstress_" + d + "/val/*.JPG"):
				img = cv2.imread(img_path)	
				start_time = time.time()
				predictions, vis_img = run_on_image(predictor, img)
				end_time = time.time()
				logger.info(
                				"{}: {} in {:.2f}s".format(
                    				img_path,
                    				"detected {} instances".format(len(predictions["instances"]))
                    				if "instances" in predictions
                    				else "finished",
                    				end_time - start_time,
                				)
            			)
				save_path = args.output + "/threshold_" + str(i) + "/" + os.path.basename(img_path)
				vis_img.save(save_path)
				total_time += end_time - start_time
				total_imgs += 1
			avg_time = total_time / total_imgs
			print("Average Inferencing Time for {} dataset: {:.2f}s".format("roadstress_" + d,avg_time))
		
		# CoCo Format Performance Evaluation:
		model = build_model(cfg)
		DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
		do_test(cfg, model)
		print("-------------------------------------------------------------------")
