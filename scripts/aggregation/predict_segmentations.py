import os
from glob import glob
import pickle5 as pickle
import argparse
from easydict import EasyDict
from tqdm import tqdm

import torch
import cv2

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures.instances import Instances
from detectron2.projects import point_rend

from constants.datasets import LVIS_METADATA_PTH
from constants.ldm import DEFAULT_LDM_MODEL_KEY
from constants.segmentation import COCO_SEG_CONFIG_PTH, COCO_SEG_WEIGHTS_PTH, LVIS_SEG_CONFIG_PTH, LVIS_SEG_WEIGHTS_PTH, DEFAULT_SEGMENTATION_THRESHOLD
from constants.metadata import DEFAULT_IMAGE_DIR, DEFAULT_SEGMENTATION_DIR, DEFAULT_SEED

from utils.prepare_prompts import prepare_cps_from_dirs
from utils.reproducibility import seed_everything




def segmentation_coco(
        cps, 
        ldm_model_key,
        image_dir, 
        save_dir,
        thres, 
        skip_done, 
        verbose,
    ):

    # make arguments as easydict
    hparams = EasyDict(dict(threshold=thres))

    # setup coco metadata
    setup_logger()
    coco_metadata = MetadataCatalog.get("coco_2017_val")

    # get segmentation model (coco)
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg) # --> Add PointRend-specific config
    cfg.merge_from_file(COCO_SEG_CONFIG_PTH)
    cfg.MODEL.WEIGHTS = COCO_SEG_WEIGHTS_PTH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = hparams.threshold
    cfg.MODEL.DEVICE = "cuda"

    # get segmentation model
    SegmentationModel=DefaultPredictor(cfg)
    
    # clear screen if not verbose
    if not verbose: os.system("clear")
    
    # run segmentation for images
    for category, prompt in cps:
        # all image paths
        img_pths = sorted(list(glob(f"./{image_dir}/{ldm_model_key}/{category}/{prompt}/*/*.png")))
        
        # iterate through image path
        for img_pth in tqdm(img_pths, desc=f"Predicting Segmentations for [Category '{category}' / Prompt '{prompt}']"):
            if verbose: print("Processing: ", img_pth)
            
            # augprompt
            augprompt = img_pth.split("/")[-2]
            # Image ID
            _id, _ext = img_pth.split("/")[-1].split(".")
            # Create result-save directory
            result_save_dir = f"./{save_dir}/{ldm_model_key}/{category}/{prompt}/{augprompt}"
            os.makedirs(result_save_dir, exist_ok=True)

            # save path (segmentation results, visualized image)
            img_save_pth = os.path.join(result_save_dir, f"{_id}.{_ext}")
            seg_save_pth = os.path.join(result_save_dir, f"{_id}.pickle")
            
            # skip if exists
            if os.path.exists(img_save_pth) and os.path.exists(seg_save_pth) and skip_done:
                if verbose: print("Already processed: ", img_save_pth)
                continue
    
            # read image
            im=cv2.imread(img_pth)

            # run segmentation
            outputs=SegmentationModel(im)
            instances=outputs["instances"]

            # visualize segmentation on image
            v = Visualizer(im[:,:,::-1], coco_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
            rend_result=v.draw_instance_predictions(instances.to("cpu")).get_image()
            
            # save visualized image
            cv2.imwrite(img_save_pth, rend_result[:,:,::-1])
            
            # save segmentation information
            with open(seg_save_pth, "wb") as handle:
                pickle.dump(instances.to("cpu"), handle, protocol=pickle.HIGHEST_PROTOCOL)






def segmentation_lvis(
        cps,
        ldm_model_key,
        image_dir, 
        save_dir,
        thres, 
        skip_done, 
        verbose,
    ):

    # make arguments as easydict
    hparams = EasyDict(dict(threshold=thres))

    # setup coco metadata
    setup_logger()
    coco_metadata = MetadataCatalog.get("coco_2017_val")

    # setup lvis metadata
    with open(LVIS_METADATA_PTH, "rb") as handle:
        lvis_metadata = pickle.load(handle)

    # get segmentation model (lvis)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(LVIS_SEG_CONFIG_PTH))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(LVIS_SEG_WEIGHTS_PTH)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = hparams.threshold
    cfg.MODEL.DEVICE = "cuda"

    # get segmentation model
    SegmentationModel=DefaultPredictor(cfg)
    
    # get segmentation model (coco) to segment human only
    cfg_coco = get_cfg()
    point_rend.add_pointrend_config(cfg_coco) # --> Add PointRend-specific config
    cfg_coco.merge_from_file(COCO_SEG_CONFIG_PTH)
    cfg_coco.MODEL.WEIGHTS = COCO_SEG_WEIGHTS_PTH
    cfg_coco.MODEL.ROI_HEADS.SCORE_THRESH_TEST = hparams.threshold
    cfg_coco.MODEL.DEVICE = "cuda"

    # get segmentation model (only for human)
    HumanSegmentationModel=DefaultPredictor(cfg_coco)

    # clear screen if not verbose
    if not verbose: os.system("clear")

    # run segmentation for images
    for category, prompt in cps:
        # all image paths
        img_pths = sorted(list(glob(f"./{image_dir}/{ldm_model_key}/{category}/{prompt}/*/*.png")))
        
        # iterate through image path
        for img_pth in tqdm(img_pths, desc=f"Category: {category} / Prompt: {prompt}"):
            if verbose: print("Processing: ", img_pth)
            # augprompt
            augprompt = img_pth.split("/")[-2]
            # Image ID
            _id, _ext = img_pth.split("/")[-1].split(".")
            # Create result-save directory
            result_save_dir = f"./{save_dir}/{ldm_model_key}/{category}/{prompt}/{augprompt}"
            os.makedirs(result_save_dir, exist_ok=True)

            # save path (segmentation results, visualized image)
            img_save_pth = os.path.join(result_save_dir, f"{_id}.{_ext}")
            seg_save_pth = os.path.join(result_save_dir, f"{_id}.pickle")
            
            # skip if exists
            if os.path.exists(img_save_pth) and os.path.exists(seg_save_pth) and skip_done:
                if verbose: print("Already processed: ", img_save_pth)
                continue
    
            # read image
            im=cv2.imread(img_pth)

            # run segmentation
            instances_lvis=SegmentationModel(im)['instances'] # for object
            instances_coco=HumanSegmentationModel(im)['instances'] 
            instances_coco=instances_coco[instances_coco.pred_classes == 0] # for human
            instances = Instances.cat([instances_lvis, instances_coco]).to('cpu')
            
            # indices for object (lvis)
            is_lvis = torch.tensor([True] * len(instances_lvis) + [False] * len(instances_coco))
            instances.is_lvis = is_lvis

            # visualize segmentation on image
            v = Visualizer(im[:,:,::-1], lvis_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
            rend_result=v.draw_instance_predictions(instances.to("cpu")).get_image()
            
            # save visualized image
            cv2.imwrite(img_save_pth, rend_result[:,:,::-1])
            
            # save segmentation information
            with open(seg_save_pth, "wb") as handle:
                pickle.dump(instances, handle, protocol=pickle.HIGHEST_PROTOCOL)


def segmentation(mode, **kwargs):
    assert mode in ["coco", "lvis"], f"Segmentation Mode: {mode} --> Not implemented..."
    # coco categories
    if mode == "coco":
        segmentation_coco(**kwargs)
    # lvis categories
    if mode == "lvis":
        segmentation_lvis(**kwargs)


if __name__ == "__main__":
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--ldm_model_key", type=str, default=DEFAULT_LDM_MODEL_KEY)

    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SEGMENTATION_DIR)

    parser.add_argument("--mode", type=str, choices=["coco", "lvis"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_SEGMENTATION_THRESHOLD)

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # load sorted (category / prompt) pairs
    cps = prepare_cps_from_dirs(args, image_dir=args.image_dir, use_filter_setting=False)

    # seed for reproducible generation
    seed_everything(args.seed)

    # print information
    if args.verbose:
        print("===Arguments===")
        print(args)

    # run segmentation
    segmentation(
        mode=args.mode,
        cps=cps, 
        ldm_model_key=args.ldm_model_key,
        image_dir=args.image_dir, 
        save_dir=args.save_dir,
        thres=args.threshold, 
        skip_done=args.skip_done, 
        verbose=args.verbose,
    )