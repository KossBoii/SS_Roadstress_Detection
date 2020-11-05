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
def config(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.backbone))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.backbone)    

    # dataset configuration  
    cfg.DATASETS.TRAIN = (args.training_dataset + "_train",)
    cfg.DATASETS.TEST = ("roadstress_new_val", "roadstress_old_val")
    
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2                    # 2 GPUs --> each GPU will see 1 image per batch
    cfg.SOLVER.WARMUP_ITERS = 2000                  # 
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 150000
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,8,16,32,64,128]]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024	# 1024
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1             # 1 category (roadway stress)

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]]	# [[0.25, 0.5, 1.0, 2.0, 4.0, 8.0]]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.7
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 800

    #cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    #cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
    #cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    #cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10000

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
    for d in ["train", "val"]:
        DatasetCatalog.register(args.training_dataset + "_" + d, lambda d=d: get_roadstress_dicts("dataset/semi_sup_learning/" + args.training_dataset + "/" + d))
        MetadataCatalog.get(args.training_dataset + "_" + d).set(thing_classes=["roadstress"])            # specify the category names
        MetadataCatalog.get(args.training_dataset + "_" + d).set(evaluator_type="coco")                   # coco evaluator

    name=""
    if args.training_dataset == "roadstress_new":
        name = "roadstress_old_val"
    elif args.training_dataset == "roadstress_old":
        name = "roadstress_new_val"
    
    DatasetCatalog.register(name, lambda: get_roadstress_dicts("dataset/semi_sup_learning/" + name[:-4] + "/val"))
    MetadataCatalog.get(name).set(thing_classes=["roadstress"])            # specify the category names
    MetadataCatalog.get(name).set(evaluator_type="coco")                   # coco evaluator
    print("Done Registering the dataset")

    # Getting the metadata for the roadstress dataset    
    roadstress_metadata = MetadataCatalog.get("roadstress_new_val")
    cfg = config(args)                      # setup the config from the cmd arguments
    
    #----------------------------------------- Custom Training Loop -----------------------------------------------------
    # model = build_model(cfg)                # build the model architecture based on the config
    # logger.info("Model:\n{}".format(model))
    # if args.eval_only:
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     print("Eval only: True")
    #     return do_test(cfg, model)

    # distributed = comm.get_world_size() > 1
    # if distributed:
    #     model = DistributedDataParallel(
    #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
    #     )

    # do_train(cfg, model, resume=args.resume)
    # print("Finish training the model. Run inference and evaluation!")
    # return do_test(cfg, model)

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
    parser.add_argument("--training-dataset", required=True, help="dataset name to train")
    parser.add_argument("--backbone", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="backbone model")

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