import os
from glob import glob
import argparse

import pickle5 as pickle
from tqdm import tqdm
from easydict import EasyDict

import math
import numpy as np
import torch
import torch.nn as nn

import cv2
import matplotlib.pyplot as plt

from constants.ldm import DEFAULT_LDM_MODEL_KEY
from constants.filtering import DEFAULT_FILTER_SETTING_NUM
from constants.perspective import DEFAULT_FOV, DEFAULT_STOP_THRES, DEFAULT_LR, DEFAULT_EPOCHS
from constants.metadata import DEFAULT_FILTERING_DIR, DEFAULT_PERSPECTIE_CAMERA_DIR, DEFAULT_SEED

from utils.prepare_prompts import prepare_cps_from_dirs
from utils.reproducibility import seed_everything
from utils.settings import load_filter_settings
from utils.perspective_conversion import apply_wp_to_p


def convert_to_perspective(cps, args):
    # clear screen if not verbose
    if not args.verbose: os.system("clear")

    ## iterate for all setting numbers
    for filter_setting_num in args.filter_setting_nums:
        # load filter_settings to arguments
        filter_settings = load_filter_settings(args, filter_setting_num)
    
        # run 'weak perspective->perspective' for all (category, prompt) pairs
        for category, prompt in cps:
            # all filtered images path
            filtered_img_pths = sorted(list(glob(f"./{filter_settings['filter_dir']}/{filter_settings['ldm_model_key']}/{category}/settings:{filter_setting_num:03}/{prompt}/*.png")))

            # all filtered image path
            for filtered_img_pth in tqdm(filtered_img_pths, desc=f"Converting WP->P for [Category '{category}' / Prompt '{prompt}' / Setting '{filter_setting_num:03}']"):
                # print
                if filter_settings['verbose']:
                    print("Processing: ", filtered_img_pth)

                # augprompt
                augprompt = filtered_img_pth.split("/")[-1].split(":")[0]
                # image id
                _id, _ext = filtered_img_pth.split("/")[-1].split(":")[1].split(".")
                # result-save directory
                result_save_dir = f"./{filter_settings['save_dir']}/{filter_settings['ldm_model_key']}/{category}/{prompt}"
                os.makedirs(result_save_dir, exist_ok=True)
                # result-save paths
                cam_save_pth = f"{result_save_dir}/{augprompt}:{_id}.pickle"

                # image pth / frankmocap pth
                img_pth = f"./{filter_settings['image_dir']}/{filter_settings['ldm_model_key']}/{category}/{prompt}/{augprompt}/{_id}.png"
                human_pth = f"./{filter_settings['human_dir']}/{filter_settings['ldm_model_key']}/{category}/{prompt}/{augprompt}/{_id}.pickle"
                
                # skip if already done
                if os.path.exists(cam_save_pth) and filter_settings['skip_done']:
                    if filter_settings['verbose']:
                        print("\tContinueing since ALREADY DONE!!", cam_save_pth)
                    continue

                ## convert 'weak perspective->perspective'
                persp_cam_list, human_parameters = apply_wp_to_p(
                    filter_settings=filter_settings, 
                    img_pth=img_pth, 
                    human_pth=human_pth,
                    mocap_output_list=None
                )
                
                # save perspective camera
                with open(cam_save_pth, "wb") as handle:
                    if args.save_with_human_params:
                        tosave = {
                            'human_parameters': human_parameters,
                            'persp_cam_list': persp_cam_list
                        }
                        pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        tosave = {
                            'persp_cam_list': persp_cam_list
                        }
                        pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    ## Arguments
    parser = argparse.ArgumentParser()
    # important
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--ldm_model_key", type=str, default=DEFAULT_LDM_MODEL_KEY)
    parser.add_argument("--filter_setting_nums", nargs="+", type=int, default=[DEFAULT_FILTER_SETTING_NUM])

    parser.add_argument("--filter_dir", type=str, default=DEFAULT_FILTERING_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_PERSPECTIE_CAMERA_DIR)

    parser.add_argument("--use_openpose_joints", action="store_true")
    parser.add_argument("--ignore_joints_outside", action="store_true")
    parser.add_argument("--freeze_foc", action="store_true")
    parser.add_argument("--fovy", type=float, nargs="?", default=DEFAULT_FOV)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--thres", type=float, default=DEFAULT_STOP_THRES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--save_with_human_params", action="store_true")

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # load sorted (category / prompt) pairs
    cps = prepare_cps_from_dirs(args, image_dir=args.filter_dir, use_filter_setting=True)

    # seed for reproducible generation
    seed_everything(args.seed)
    
    # arguments
    if args.verbose:
        print("===Arguments===")
        print(args)
    
    # weak perspective -> perspective
    convert_to_perspective(cps=cps, args=args)