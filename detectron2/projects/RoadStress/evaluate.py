from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from utils import *
# logger = logging.getLogger("detectron2")

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

def create_dt(input, output, map):
    img_size = output[0]["instances"].image_size
    # pred_boxes = outputs[0]["instances"].get('pred_boxes').tensor.tolist()
    pred_boxes = np.round(output[0]["instances"].get('pred_boxes').tensor.cpu().detach().numpy()).tolist()
    scores = output[0]["instances"].get('scores').tolist()
    classes = output[0]["instances"].get('pred_classes').tolist()
    masks = output[0]["instances"].get('pred_masks').cpu().data.numpy()
    count = len(output[0]["instances"])

    with open(input[0]["file_name"][-12:-4] + ".txt", 'w') as f:
      for i in range(0, count):
        f.write("%s %f %d %d %d %d\n" % (
            map[classes[i]],
            scores[i],
            pred_boxes[i][0],
            pred_boxes[i][1],
            pred_boxes[i][2],
            pred_boxes[i][3],
        ))

def create_gt(json_file_path, save_path):
    # Load and read json file stores information about annotations
    with open(json_file_path) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []          # list of annotations info for every images in the dataset
    for idx, v in enumerate(imgs_anns.values()):
        save_file = os.path.join(save_path, v["filename"][:-4] + ".txt")
        with open(save_file, "w") as f1:
            # getting annotation for every instances of object in the image
            annos = v["regions"]
            objs = []
            for anno in annos:
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]

                f1.write("%s %d %d %d %d\n" % (
                    "roadstress",
                    np.min(px),
                    np.min(py),
                    np.max(px),
                    np.max(py),
                ))

if __name__ == "__main__":
    # create_gt("dataset/roadstress_new/train/via_export_json.json", "dataset/roadstress_new/")
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

    for i in np.arange(0.05, 0.95, 0.05):
        i = i.round(decimals=2)
        print("\n\n-------------------------------------------------------------------")
        print("Threshold: " + str(i))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(i)
        
        for d in ["old", "new"]:
            # create image folder
            os.makedirs(os.path.join(args.output, "%s/threshold_%.2f" % ("roadstress_" + d + "_val", i)), exist_ok=True)

            data_loader = build_detection_test_loader(cfg, "roadstress_" + d + "_val", mapper=DatasetMapper(cfg, False))
            model = build_model(cfg)
            DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
            model.eval()

            map = {
                0: "roadstress"
            }

            for input in data_loader:
                output = model(input)

                pred_boxes = np.round(output[0]["instances"].get('pred_boxes').tensor.cpu().detach().numpy()).tolist()
                scores = output[0]["instances"].get('scores').tolist()
                classes = output[0]["instances"].get('pred_classes').tolist()
                count = len(output[0]["instances"])

                print(args.output + "/%s/threshold_%.2f/%s" % 
                    (
                        "roadstress_" + d + "_val", 
                        i,
                        input[0]["file_name"][-12:-4] + ".txt"
                    )
                )

                with open(
                    args.output + "/%s/threshold_%.2f/%s" % 
                        (
                            "roadstress_" + d + "_val", 
                            i,
                            input[0]["file_name"][-12:-4] + ".txt"
                        )
                    , "w"
                ) as f:
                    for j in range(0, count):
                        f.write("%s %f %d %d %d %d\n" % (
                            map[classes[j]],
                            scores[j],
                            pred_boxes[j][0],
                            pred_boxes[j][1],
                            pred_boxes[j][2],
                            pred_boxes[j][3],
                        ))

    # map = {
    #     0: "roadstress"
    # }
    # for idx, input in enumerate(data_loader):
    #     output = model(input)
    #     create_dt(input, output, map)