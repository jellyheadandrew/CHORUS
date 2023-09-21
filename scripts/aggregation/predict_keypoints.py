import os
from glob import glob
import argparse
import pickle5 as pickle
from tqdm import tqdm

import numpy as np
import cv2

from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result, process_mmdet_results
from mmdet.apis import inference_detector, init_detector

from constants.ldm import DEFAULT_LDM_MODEL_KEY
from constants.keypoint_detection import MMPOSE_DETECTOR_SETTINGS, DEFAULT_KEYPOINT_THRESHOLD, DEFAULT_KEYPOINT_BBOX_THRESHOLD, DEFAULT_MMPOSE_DETECTOR_KEY
from constants.metadata import DEFAULT_IMAGE_DIR, DEFAULT_HUMAN_DIR, DEFAULT_KEYPOINT_DIR, DEFAULT_SEED

from utils.prepare_prompts import prepare_cps_from_dirs
from utils.reproducibility import seed_everything


def keypoint_prediction(
        cps,
        ldm_model_key,
        thres, # only for visualization
        skip_done,
        verbose,
        image_dir,
        human_dir, 
        save_dir,
        bbox_threshold,
        pose_config,
        pose_checkpoint,
    ):
    # load pretrained pose model
    pose_model = init_pose_model(pose_config, pose_checkpoint, device = "cuda")
    
    # clear screen if not verbose
    if not verbose: os.system("clear")
    
    # run keypoint prediction for all (category, prompt) pairs
    for category, prompt in cps:
        # all image path
        img_pths = sorted(list(glob(f"./{image_dir}/{ldm_model_key}/{category}/{prompt}/*/*.png")))
        for img_pth in tqdm(img_pths, desc=f"Predicting Keypoint Filters for [Category '{category}' / Prompt '{prompt}']"):
            # print
            if verbose:
                print("Processing: ", img_pth)

            # augprompt
            augprompt = img_pth.split("/")[-2]
            # image id
            _id, _ext = img_pth.split("/")[-1].split(".")
            # result-save directory
            result_save_dir = f"./{save_dir}/{ldm_model_key}/{category}/{prompt}/{augprompt}"
            os.makedirs(result_save_dir, exist_ok=True)
            # result-save paths
            img_save_pth = os.path.join(result_save_dir, f"{_id}.png")
            keypoint_save_pth = os.path.join(result_save_dir, f"{_id}.pickle")
    
            # skip if already done
            if os.path.exists(img_save_pth) and os.path.exists(keypoint_save_pth) and skip_done:
                if verbose:
                    print("\tContinueing since ALREADY DONE!!", keypoint_save_pth)
                continue
        
            # load human params
            human_pth = f"./{human_dir}/{ldm_model_key}/{category}/{prompt}/{augprompt}/{_id}.pickle"
            with open(human_pth, 'rb') as handle:
                human_parameters = pickle.load(handle)

            # if there is no human, return empty list
            if human_parameters == "NO HUMANS":
                if verbose:
                    print(f"\tNo humans for --> {human_pth}")
                person_results = []
            # if there is human, return bbox-list & confidence-list 
            else:
                # bbox-list / confidence-list
                body_bbox_list = human_parameters['body_bbox_list_xyxy'] # (shape: Nx4, xyxy)
                confidence_list = human_parameters['confidence_list'] # (shape: N)

                # conversion to mmdet format (Kx3, last column is confidence)
                bbox_conf_arr = np.concatenate([body_bbox_list, np.expand_dims(confidence_list, axis=-1)], axis=-1)
                person_results = [None] * len(bbox_conf_arr)
                # stack to list for all people
                for i in range(len(bbox_conf_arr)):
                    person_results[i] = {'bbox': bbox_conf_arr[i]}
            

            # inference pose
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img_pth,
                person_results,
                bbox_thr=bbox_threshold, # 0.0
                format='xyxy',
                dataset=pose_model.cfg.data.test.type
            )

            # render keypoint prediction to image
            vis_result = vis_pose_result(
                pose_model,
                img_pth,
                pose_results,
                kpt_score_thr=thres,
                dataset=pose_model.cfg.data.test.type,
                show=False
            )

            # save rendered image
            cv2.imwrite(img_save_pth, vis_result)
            # save keypoint prediction (as numpy file)
            with open(keypoint_save_pth, "wb") as handle:
                pickle.dump(pose_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--ldm_model_key", type=str, default=DEFAULT_LDM_MODEL_KEY)

    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--human_dir", type=str, default=DEFAULT_HUMAN_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_KEYPOINT_DIR)
        
    parser.add_argument("--detector_setting",
        nargs="?",
        choices=sorted(list(MMPOSE_DETECTOR_SETTINGS.keys())),
        default=DEFAULT_MMPOSE_DETECTOR_KEY
    )
    parser.add_argument("--thres", type=float, default=DEFAULT_KEYPOINT_THRESHOLD) # keypoint confidence threshold for visualization
    parser.add_argument("--bbox_threshold", type=float, default=DEFAULT_KEYPOINT_BBOX_THRESHOLD) # bbox where [confidence>bbox_threshold] used only
    
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # load sorted (category / prompt) pairs
    cps = prepare_cps_from_dirs(args, image_dir=args.image_dir, use_filter_setting=False)

    # seed for reproducible generation
    seed_everything(args.seed)

    # arguments
    if args.verbose:
        print("===Arguments===")
        print(args)
    
    # predict 2d keypoints
    keypoint_prediction(
        cps=cps,
        ldm_model_key=args.ldm_model_key,
        thres=args.thres, 
        skip_done=args.skip_done,
        verbose=args.verbose,
        image_dir=args.image_dir,
        human_dir=args.human_dir, 
        save_dir=args.save_dir,
        bbox_threshold=args.bbox_threshold,
        pose_config=MMPOSE_DETECTOR_SETTINGS[args.detector_setting][0],
        pose_checkpoint=MMPOSE_DETECTOR_SETTINGS[args.detector_setting][1],
    )