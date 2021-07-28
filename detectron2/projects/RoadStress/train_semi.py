from utils import *
import sys
logger = logging.getLogger("detectron2")
import random
from process_annos import combine_annos

model_id_to_backbone = {
    '07102020155403': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    '07112020084228': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    '07102020155420': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
    '07112020103104': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
    '07102020155432': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
    '07112020170550': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
}

'''
    Backbone model:
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

        - "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        - "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" 

''' 
def get_roadstress_dicts_modified(img_dir, anno_json_name):
    # Load and read json file stores information about annotations
    json_file = './pseudo/' + anno_json_name
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []          # list of annotations info for every images in the dataset
    for idx, v in enumerate(imgs_anns.values()):
        if(v["regions"]):
            record = {}         # a dictionary to store all necessary info of each image in the dataset
            
            # open the image to get the height and width
            filename = os.path.join(img_dir, v["filename"])
            print(filename)
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
          
            # getting annotation for every instances of object in the image
            annos = v["regions"]
            objs = []
            for anno in annos:
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def config(args, model_id):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_id_to_backbone[model_id]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id_to_backbone[model_id])    

    # dataset configuration  
    cfg.DATASETS.TRAIN = (args.training_dataset,)
    
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 2                    # 2 GPUs --> each GPU will see 1 image per batch
    cfg.SOLVER.WARMUP_ITERS = 2000                  # 
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,16,32,64,128]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128	# 1024
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             # 1 category (roadway stress)

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0]]	# [[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.7
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 800

    # Setup Logging folder
    curTime = datetime.now()
    cfg.OUTPUT_DIR = "./output/" + curTime.strftime("%m%d%Y%H%M%S")
    if not os.path.exists(os.getcwd() + "/output/"):
        os.makedirs(os.getcwd() + "/output/", exist_ok=True)
        print("Done creating folder output!")
    else:
        print("Folder output/ existed!")
    if not os.path.exists(os.getcwd() + "/output/" + curTime.strftime("%m%d%Y%H%M%S")):    
        os.makedirs(os.getcwd() + "/output/" + curTime.strftime("%m%d%Y%H%M%S"), exist_ok=True)

    cfg.freeze()                    # make the configuration unchangeable during the training process
    default_setup(cfg, args)
    return cfg

class Trainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(cfg, mapper=customMapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def run_on_image(predictor, img_path):
    img = cv2.imread(img_path)

    # compute prediction
    predictions = predictor(img)

    dicts = {}
    if "instances" in predictions:
        instances = predictions["instances"].to(torch.device("cpu"))
        count = len(instances)

        # Get binary mask for each instances
        bin_masks = instances.get("pred_masks").detach().numpy()
        dicts["filename"] = img_path[-12:]
        dicts["size"] = random.randint(1000000, 9999999)

        regions = []
        for i in range(0,len(instances)):
            bin_mask = bin_masks[i]
                    
            # convert binary mask to polygon
            pairs = mask_to_poly(bin_mask)
            x_pts = []
            y_pts = []

            if len(pairs) == 0 or len(pairs[0]) == 0:
                count = count - 1
                continue
                    
            for j in range(0, len(pairs[0]), 2):
                x_pts.append(pairs[0][j])

            for j in range(1, len(pairs[0]), 2):
                y_pts.append(pairs[0][j])

            region = {}
            temp = {}
            # shape_attributes
            temp['name'] = 'polygon'
            temp['all_points_x'] = x_pts
            temp['all_points_y'] = y_pts
            region['shape_attributes'] = temp

            # region_attributes
            region['region_attributes'] = {
                'name': 'roadstress'
            }
            regions.append(region)
    
    dicts['regions'] = region
    dicts['file_attributes'] = {}

    return dicts, count

def main(args):
    # get manually annotated images
    man_anno_imgs = []
    for img_path in glob.iglob('./dataset/train/'+ args.training_dataset + "/*.JPG"):
        file_name = img_path[-12:]
        man_anno_imgs.append(file_name)

    # Load cfg
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join('./output/' + args.model_id, 'config.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.WEIGHTS = os.path.join('./output/' + args.model_id, 'model_final.pth')
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 5000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 5000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 5000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000

    # Load model and checkpoint
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
    predictor = DefaultPredictor(cfg)
    
    TRAINING_SAMPLE_SIZE = 20
    pseudo_ratios = [0.25, 0.5, 0.75]

    for pseudo_num in pseudo_ratios:
        num_imgs_pseudo = int(TRAINING_SAMPLE_SIZE / (1 - pseudo_num) * pseudo_num)

        # Compute pseudo-label
        count = 0
        result_dicts = {}

        for img_path in glob.iglob('./dataset/pseudo/'+ args.training_dataset + "/*.JPG"):
            if img_path[-12:] in man_anno_imgs:
                continue
            if count == num_imgs_pseudo:
                break
            if count < num_imgs_pseudo:
                result, instance_count = run_on_image(predictor, img_path)

                result_dicts[result["filename"] + "." + str(result["size"])] = result
                print("Finish generate predictions for %s. Predict %d instances" % (result["filename"], instance_count))

                # save the pseudo-label annotation file
                with open('./pseudo/pseudoLabel_' + args.model_id + '_' + str(num_imgs_pseudo) + '.json', "w") as f:
                    json.dump(result_dicts, f)

        # Combine pseudo-label with original annotation file
        com_annos_filename = combine_annos('.dataset/train/' + args.training_dataset + '/via_export_json.json', 
                        './pseudo/pseudoLabel_' + args.model_id + '_' + str(num_imgs_pseudo) + '.json', 
                        args.model_id
        )

        #----------------------------------------- Trainer Training Loop ----------------------------------------------------
        # Register the dataset:
        for d in ["train"]:
            DatasetCatalog.register(args.training_dataset , lambda: get_roadstress_dicts_modified(
                '.dataset/pseudo/' + args.training_dataset, com_annos_filename)
            )
            MetadataCatalog.get(args.training_dataset).set(thing_classes=["roadstress"])            # specify the category names
            MetadataCatalog.get(args.training_dataset).set(evaluator_type="coco")                   # coco evaluator
        print("Done Registering the dataset")

        cfg = config(args, args.model_id)
        cfg.MODEL.WEIGHTS = os.path.join('./output/' + args.model_id, 'model_final.pth')

        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        print("Finish training the model. Run inference and evaluation!")
        do_test(cfg, trainer.model)

def custom_default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--training-dataset", required=True, help="dataset name to train")
    parser.add_argument("--model_id", required=True, help="model id from beginning")
    # parser.add_argument("--backbone", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="backbone model")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    args = custom_default_argument_parser().parse_args()
    print("Command Line Args:", args)
    torch.backends.cudnn.benchmark = True
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )