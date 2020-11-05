from utils import * 
import random
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
import torch, torchvision

logger = logging.getLogger("detectron2")

def config(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)

	# Set score_threshold for builtin models
	# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
	# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
	# cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.40
	cfg.TEST.DETECTIONS_PER_IMAGE = 500
	cfg.MODEL.WEIGHTS = args.weight
	cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000  # originally 1000
	cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 5000  # originally 1000
	cfg.freeze()
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

def run_on_image(predictor, img_path):
	img = cv2.imread(img_path)
	predictions = predictor(img)
	
	if "instances" in predictions:
		instances = predictions["instances"].to(torch.device("cpu"))
		count = len(instances)

		# Get binary mask for each instances
		bin_masks = instances.get("pred_masks").detach().numpy()
		dicts = {}
		dicts["filename"] = img_path[-12:]
		dicts["size"] = random.randint(1000000, 9999999)

		# print("Shape: " + str(bin_masks.shape))

		regions = []
		for i in range(0,len(instances)):
			bin_mask = bin_masks[i]

			pairs = mask_to_poly(bin_mask)
			x_pts = []
			y_pts = []		
			# print("Instance %d" % i)
			# print(pairs)
			
			if len(pairs) == 0 or len(pairs[0]) == 0:
				# There will be cases where there is bounding box,
				# 	but the model couldn't figure out segmentation mask, then skip those instances 
				count = count - 1
				continue

			for j in range(0, len(pairs[0]), 2):
				# print(j)
				x_pts.append(pairs[0][j]) 
				
			for j in range(1, len(pairs[0]), 2):
				# print(j)
				y_pts.append(pairs[0][j]) 

			region = {}
			temp = {}
			temp["name"] = "polyline"
			temp["all_points_x"] = x_pts
			temp["all_points_y"] = y_pts

			region["shape_attributes"] = temp
			region["region_attributes"] = {
				"name": "roadstress"
			}
			regions.append(region)

		dicts["regions"] = regions
		dicts["file_attributes"] = {}

		return dicts, count
	else:
		print("Something wrong. Please check the inference.py script")

if __name__ == "__main__":
	args = get_parser().parse_args()
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	# Register the dataset:
	for d in ["train", "val"]:
		DatasetCatalog.register("roadstress_" + d, lambda d=d: get_roadstress_dicts("dataset/roadstress_new/" + d))
		MetadataCatalog.get("roadstress_" + d).set(thing_classes=["roadstress"])
		MetadataCatalog.get("roadstress_" + d).set(evaluator_type="coco")
		roadstress_metadata = MetadataCatalog.get("roadstress_train")
	print("Done Registering the dataset")


	cfg = config(args)
	print(cfg.dump())
	print("Finish Setting Up Config")

	predictor = DefaultPredictor(cfg)
	print("Start Inferencing")
	# for img_path in glob.iglob(args.dataset + "/*.JPG"):
	# 	img = cv2.imread(img_path)	
	# 	start_time = time.time()
	# 	predictions, vis_img = run_on_image(predictor, img)
	# 	logger.info(
    #             	"{}: {} in {:.2f}s".format(
    #                 		img_path,
    #                 		"detected {} instances".format(len(predictions["instances"]))
    #                 		if "instances" in predictions
    #                 		else "finished",
    #                 		time.time() - start_time,
    #             	)
    #         	)
	
	# model = build_model(cfg)
	# DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)

	result_dicts = {}
	count = 0

	for img_path in glob.iglob(args.dataset + "/*.JPG"):
		result, count = run_on_image(predictor, img_path)
		result_dicts[result["filename"] + "." + str(result["size"])] = result
		print("Finish generate predictions for %s. Predict %d instances" % (result["filename"], count))

	with open("pseudoLabel.json", "w") as f:
		json.dump(result_dicts, f)

	# result = run_on_image(predictor, args.dataset + "DJI_0001.JPG")
	# print(result)




