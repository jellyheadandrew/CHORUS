import os
from glob import glob
from copy import deepcopy
import argparse
import pickle5 as pickle
import json
from tqdm import tqdm

import torch
import numpy as np

import cv2
import matplotlib.pyplot as plt

from constants.ldm import DEFAULT_LDM_MODEL_KEY
from constants.filtering import DEFAULT_FILTER_SETTING_NUM
from constants.aggr_settings.quant import QUANT_AGGR_SETTINGS
from constants.datasets import EXTENDED_COCO_EFT_VAL_SAVE_PTH, COCO_VAL_IMAGE_DIR
from constants.evaluation import DEFAULT_DOWNSCALE_LENGTH, DEFAULT_NUM_ENFORCE_MULTIVIEW, DEFAULT_PRECISON_THRESHOLDS
from constants.metadata import DEFAULT_TOTAL_AGGREGATION_NAME, DEFAULT_SEED, DEFAULT_AGGREGATION_DIR, DEFAULT_QUANT_EVAL_DIR

from utils.reproducibility import seed_everything
from utils.prepare_prompts import prepare_cps_from_dirs, get_unique_categories
from utils.misc import load_category_id
from utils.settings import aggr_save_name
from utils.aggregation.aggregator import Aggregator
from utils.aggregation.core import prepare_aggr_core
from utils.evaluation import compute_pap, compute_mean_pap



def evaluate_pap(
        cps,
        ldm_model_key,
        aggr_dir,
        save_dir,
        aggr_settings,
        filter_setting_comb,
        eft_dataset_pth,
        num_multiview_thres,
        specified_image_ids,
        downscale_length,
        thresholds,
        skip_done,
        verbose
    ):
    # print progress
    print("Loading COCO-EFT Dataset for Quantitative Evaluation...")
    
    ## load extended COCO-EFT data
    # open file
    with open(eft_dataset_pth, "rb") as handle: test_dataset = pickle.load(handle)
    # data-to-use
    all_test_data = test_dataset['eft_data']
    
    # use statistics & only use categories with number-of-images over 'args.num_multiview_thres'
    stats = test_dataset['stats']

    ## load metadata of aggregation-results
    # 'aggregation-results' save name
    aggr_result_save_name =  aggr_save_name(
        filter_setting_comb=filter_setting_comb, 
        aggr_setting_name=aggr_settings['NAME']
    )

    ## load aggregation settings and prepare 'aggregation_core' --> overwritten with pretrained-results later
    print("Preparing Aggregator Core...")
    aggr_core = prepare_aggr_core(verbose=verbose, **aggr_settings)

    # clear screen if verbose
    print("Preparation Complete!")
    if not verbose: os.system("clear")

    ## run evaluation for all (category, prompt) pairs
    for category, prompt in sorted(cps):
        # if number-of-images are insufficient for 'multiview evaluation', skip
        if category not in stats.keys():
            if verbose: print(f"not using category '{category}' since not in test-dataset")
            continue
        if stats[category] < num_multiview_thres:
            if verbose: print(f"not using category '{category}' since the number of images in test-dataset under {num_multiview_thres}")
            continue

        ## declare aggregator
        aggregator = Aggregator(core=deepcopy(aggr_core))
        
        ## load aggregation metadata (filter-settings, aggr-core)
        # metadata path
        metadata_pth = f"{aggr_dir}/{ldm_model_key}/{category}/{prompt}/{aggr_result_save_name}.json"
        # load metadata
        with open(metadata_pth, 'r') as rf: metadata = json.load(rf)
        # retrieve filter-settings
        filter_settings = metadata['filter_settings']

        ## load aggregation results
        # aggregation path
        aggr_pth = f"{aggr_dir}/{ldm_model_key}/{category}/{prompt}/{aggr_result_save_name}.pickle"
        # load aggregation to 'aggregator'
        aggregator.load(load_pth=aggr_pth)

        ## preapre for evaluation
        # category id
        category_id = load_category_id(category)

        # placeholder for saving quantitative-evaluation results
        evaluate_results = {
            'aggr_pth': aggr_pth,
            'category': category,
            'results': dict()
        }

        ## prepare save-path
        # save-directory and path
        quant_eval_save_dir = f"./{save_dir}/{ldm_model_key}/{category}"
        os.makedirs(quant_eval_save_dir, exist_ok=True)
        quant_eval_save_pth = f"{quant_eval_save_dir}/{prompt}:{aggr_result_save_name}.json"
        # skip if already exists
        if os.path.exists(quant_eval_save_pth) and skip_done:
            if verbose:
                print("\tContinueing since (evaluation) ALREADY DONE!!", quant_eval_save_pth)
            continue

        ## iterate for all test-eft-data
        num_done = 0
        pbar = tqdm(all_test_data, desc=f"Quantitative Evaluation for [Category '{category}' / Settings '{aggr_result_save_name}' / Num-Eval-Done ({num_done}/{stats[category]})]")
        for test_data in pbar:
            # if 'test_data' contains object category, test it
            if category in test_data['target_category_list']:
                test_data = deepcopy(test_data)
            else:
                continue
            
            # prepare input
            # placeholder for saving inputs
            _aggr_input = {}

            # load image
            img_name = test_data['imageName']
            img_id = img_name.replace(".jpg", "").replace("COCO_val2014_", "")
            img_pth = f'{COCO_VAL_IMAGE_DIR}/{img_name}'
            img = cv2.cvtColor(cv2.imread(img_pth), code=cv2.COLOR_BGR2RGB)
            
            # skip if 'specified_image_ids' did not come out
            if specified_image_ids is not None:
                if img_id not in specified_image_ids:
                    if verbose:
                        print(f"Continue '{img_id}' since 'specified_image_ids': {specified_image_ids} -> didn't come out")
                    continue 

            # load perspective camera
            persp_cam = test_data['persp_cam_list'][0]
            _aggr_input['cam_R'] = persp_cam['R']
            _aggr_input['cam_t'] = persp_cam['t']
            _aggr_input['cam_K'] = persp_cam['K']

            # load object mask
            _aggr_input['mask'] = test_data['mask'][category] # HxW
            H, W = _aggr_input['mask'].shape
            assert len(_aggr_input['mask'].shape) == 2

            # pose
            _aggr_input['pose'] = test_data['pred_aa'] # 1x72
            assert _aggr_input['pose'].shape == (1,72)

            # betas
            _aggr_input['betas'] = torch.zeros_like(test_data['pred_betas']) # 1x10
            assert _aggr_input['betas'].shape == (1,10)
            
            # preprocess
            aggr_input = aggregator.preprocess_input(cam_world_key='SMPL', **_aggr_input)
            
            # if evaluating, we must render the mask. 
            rendered_masks_per_thresh, target_mask_size = aggregator.render_discretized_distribution(
                cam_world_key='SMPL',
                cam_R = aggr_input['cam_R'],
                cam_t = aggr_input['cam_t'],
                cam_K = aggr_input['cam_K'],
                pose = aggr_input['pose'],
                betas = aggr_input['betas'],
                mask_shape = aggr_input['mask'].shape,
                downscale_length = downscale_length,
                use_normalized_K = False,
                thresholds = thresholds,
                verbose = verbose
            )
            
            # downsample target mask
            target_H, target_W = target_mask_size
            target_mask = cv2.resize(aggr_input['mask'], dsize=(target_W, target_H), interpolation=cv2.INTER_AREA)

            # compute PAP values
            PAP, PAP_strict, PAP_hoa, PAP_strict_hoa, target_area, target_area_hoa = compute_pap(
                pred=rendered_masks_per_thresh,
                target_mask=target_mask,
                verbose=verbose
            )

            # evaluate results
            evaluate_results['results'][img_name] = {
                'PAP': PAP,
                'PAP_strict': PAP_strict,
                'PAP_hoa': PAP_hoa,
                'PAP_strict_hoa': PAP_strict_hoa,
                'target_area': target_area,
                'target_area_hoa': target_area_hoa
            }

            # set description
            num_done += 1
            desc = f"Quantitative Evaluation for [Category '{category}' / Settings '{aggr_result_save_name}' / Num-Eval-Done ({num_done}/{stats[category]})]"
            pbar.set_description(desc)

        # check number-of-data used for testing
        assert num_done == stats[category]

        ## calculate 'mean-projective-average-precision'
        evaluate_results = compute_mean_pap(evaluate_results)

        ## save the evaluation results
        with open(quant_eval_save_pth, "w") as wf: json.dump(evaluate_results, wf, indent=1)


def run_quant_eval(cps, args):

    ## iterate for all 'args.aggr_setting_names'
    for aggr_setting_name in args.aggr_setting_names:
        # load aggregation setting
        aggr_settings = QUANT_AGGR_SETTINGS[aggr_setting_name]
        assert aggr_settings['NAME'] == aggr_setting_name, f"Mismatch in name for '{aggr_setting_name}'"

        # iterate for filter-settings-combinations
        for filter_setting_comb in args.filter_setting_nums:
            # print setting informations
            if args.verbose: print(f"Quant-Evaluating 'aggr_settings':{aggr_setting_name} / 'filter_setting_comb': {filter_setting_comb}")

            # filter_setting_comb as list of filter_setting_num
            filter_setting_comb = [int(i) for i in filter_setting_comb.split(",")]

            ## evaluate aggregation result with 'projective average precision' metrics and save results
            evaluate_pap(
                cps=cps,
                ldm_model_key=args.ldm_model_key,
                aggr_dir=args.aggr_dir,
                save_dir=args.save_dir,
                aggr_settings=aggr_settings,
                filter_setting_comb=filter_setting_comb,
                eft_dataset_pth=args.eft_dataset_pth,
                num_multiview_thres=args.num_multiview_thres,
                downscale_length=args.downscale_length,
                thresholds=args.thresholds,
                specified_image_ids=args.specified_image_ids,
                skip_done=args.skip_done,
                verbose=args.verbose
            )


if __name__ == "__main__":
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--prompts", nargs="+", default=[DEFAULT_TOTAL_AGGREGATION_NAME])
    parser.add_argument("--ldm_model_key", type=str, default=DEFAULT_LDM_MODEL_KEY)

    parser.add_argument("--aggr_dir", type=str, default=DEFAULT_AGGREGATION_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_QUANT_EVAL_DIR)

    parser.add_argument("--aggr_setting_names", nargs="+", type=str)
    parser.add_argument("--filter_setting_nums", nargs="+", type=str, default=[str(DEFAULT_FILTER_SETTING_NUM)]) # you can write as "--filter_settings '7,8' '9'"

    parser.add_argument("--eft_dataset_pth", type=str, default=EXTENDED_COCO_EFT_VAL_SAVE_PTH)
    parser.add_argument("--num_multiview_thres", type=int, default=DEFAULT_NUM_ENFORCE_MULTIVIEW)

    parser.add_argument("--downscale_length", type=int, default=DEFAULT_DOWNSCALE_LENGTH)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_PRECISON_THRESHOLDS)

    parser.add_argument("--specified_image_ids", nargs="+", type=str)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    # load sorted (category / prompt) pairs
    cps = prepare_cps_from_dirs(args, image_dir=args.aggr_dir, use_filter_setting=False)

    # seed for reproducible generation
    seed_everything(args.seed)

    # arguments
    if args.verbose:
        print("===Arguments===")
        print(args)

    # evaluate PAP for 'trained-results'
    run_quant_eval(cps=cps, args=args)