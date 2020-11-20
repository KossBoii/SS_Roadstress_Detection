from utils import *
import sys
logger = logging.getLogger("detectron2")

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
    json_file = os.path.join(img_dir, anno_json_name)
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


def config(args, name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.backbone))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.backbone)    

    # dataset configuration  
    cfg.DATASETS.TRAIN = (args.training_dataset,)
    
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2                    # 2 GPUs --> each GPU will see 1 image per batch
    cfg.SOLVER.WARMUP_ITERS = 2000                  # 
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,16,32,64,128]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512	# 1024
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

def main(args):
    # Register the dataset:
    for d in ["train"]:
        DatasetCatalog.register(args.training_dataset , lambda: get_roadstress_dicts_modified(
            os.path.join(args.training_path, args.training_dataset), args.json_name))
        MetadataCatalog.get(args.training_dataset).set(thing_classes=["roadstress"])            # specify the category names
        MetadataCatalog.get(args.training_dataset).set(evaluator_type="coco")                   # coco evaluator
    print("Done Registering the dataset")

    # Getting the metadata for the roadstress dataset    
    roadstress_metadata = MetadataCatalog.get(args.training_dataset)
    cfg = config(args, args.training_dataset)                      # setup the config from the cmd arguments
    cfg.dump()

    #----------------------------------------- Trainer Training Loop ----------------------------------------------------
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()    
    print("Finish training the model. Run inference and evaluation!")
    return do_test(cfg, trainer.model)

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
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--training-path", required=True, help="base path of dataset")
    parser.add_argument("--training-dataset", required=True, help="dataset name to train")
    parser.add_argument("--backbone", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="backbone model")
    parser.add_argument("--json-name", required=True, help="json file for annotations")

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