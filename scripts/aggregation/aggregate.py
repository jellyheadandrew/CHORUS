import os
from glob import glob
import argparse
import json
from copy import deepcopy
from tqdm import tqdm

import math
import random

from constants.ldm import DEFAULT_LDM_MODEL_KEY
from constants.filtering import DEFAULT_FILTER_SETTING_NUM
from constants.aggr_settings.quant import QUANT_AGGR_SETTINGS
from constants.aggr_settings.qual import QUAL_AGGR_SETTINGS
from constants.metadata import DEFAULT_TOTAL_AGGREGATION_NAME, DEFAULT_FILTERING_DIR, DEFAULT_PERSPECTIE_CAMERA_DIR, DEFAULT_AGGREGATION_DIR, DEFAULT_SEED

from utils.prepare_prompts import get_unique_categories, get_cps_for_given_category, prepare_cps_from_dirs
from utils.settings import load_filter_settings_combination, aggr_save_name
from utils.reproducibility import seed_everything
from utils.misc import load_category_id, extract_nonvector
from utils.aggregation.core import prepare_aggr_core
from utils.aggregation.load_input import load_aggr_input
from utils.aggregation.aggregator import Aggregator




def aggregate(
        cps,
        ldm_model_key,
        save_dir,
        cam_world_key,
        total_aggr_name,
        aggr_settings,
        filter_settings,
        visualize,
        skip_single_prompt,
        skip_done,
        verbose,
    ):
    # prepare aggregation core for running aggregation
    print("Preparing Aggregator Core...")

    # 'aggregation-results' save name (including 'total-aggregation')
    aggr_result_save_name = aggr_save_name(
        filter_setting_comb=filter_settings['filter_setting_comb'], 
        aggr_setting_name=aggr_settings['NAME']
    )

    # load aggregation settings and prepare 'aggregation_core' that contains all variables required, initialized
    aggr_core = prepare_aggr_core(verbose=verbose, **aggr_settings)
    
    # all unique 'augprompt:ID's per prompt, per category
    augid_per_cp = {
        category: {
            cp: sorted(list(set(
                [
                    retained_file_pth.split("/")[-1].split(".")[0] for retained_file_pth in 
                    sorted(list(glob(f"{filter_settings['filter_dir']}/{ldm_model_key}/{cp[0]}/settings:*/{cp[1]}/*.png")))
                    if int(retained_file_pth.split("/")[-3].replace("settings:","")) in filter_settings['filter_setting_comb']
                ]
            )))
            for cp in get_cps_for_given_category(cps, category)
        }
        for category in get_unique_categories(cps)
    }
    
    # if the ratio-of-data to aggregate is given, we random sample from 'augid_per_cp'
    if aggr_settings['DATA_RATIO'] < 1.:
        for category in augid_per_cp.keys(): 
            for cp in augid_per_cp[category]:
                # augprompt:id pairs
                augids = deepcopy(augid_per_cp[category][cp])
                # number of pairs to retain
                k = math.ceil(len(augids) * aggr_settings['DATA_RATIO'])
                # random-sample amount of data
                augid_per_cp[category][cp] = random.sample(augids, k=k)

    # clear screen if verbose
    print("Preparation Complete!")
    if not verbose: os.system("clear")

    ## iterate for all categories
    for category in get_unique_categories(cps):
        # (total-aggregation) result-save directory
        tot_result_save_dir = f"./{save_dir}/{ldm_model_key}/{category}/{total_aggr_name}"
        os.makedirs(tot_result_save_dir, exist_ok=True)
        # (total-aggregation) result-save paths
        tot_video_save_pth = f"{tot_result_save_dir}/{aggr_result_save_name}.mp4"
        tot_aggr_save_pth = f"{tot_result_save_dir}/{aggr_result_save_name}.pickle"
        tot_metadata_save_pth = f"{tot_result_save_dir}/{aggr_result_save_name}.json"
        
        # (total-aggregation) aggregator
        tot_aggregator = Aggregator(core=deepcopy(aggr_core))

        ## register inputs for aggregation (all prompt) 
        for cp in augid_per_cp[category].keys():
            # prompt
            prompt = cp[1]

            # result-save directory
            result_save_dir = f"./{save_dir}/{ldm_model_key}/{category}/{prompt}"
            os.makedirs(result_save_dir, exist_ok=True)
            # result-save paths
            video_save_pth = f"{result_save_dir}/{aggr_result_save_name}.mp4"
            aggr_save_pth = f"{result_save_dir}/{aggr_result_save_name}.pickle"
            metadata_save_pth = f"{result_save_dir}/{aggr_result_save_name}.json"

            # skip 'aggr' if already done (not visualize/skip-done/skip-single-prompt & aggregation/metadata exists)
            if os.path.exists(aggr_save_pth) and os.path.exists(metadata_save_pth) and os.path.exists(tot_aggr_save_pth) and os.path.exists(tot_metadata_save_pth) and (not visualize) and skip_done:
                if verbose:
                    print("\tContinueing since (aggregation) ALREADY DONE!!", tot_aggr_save_pth)
                continue
            # skip 'aggr' if already done (visualize/skip-done/skip-single-prompt & aggregation/video/metadata exists)
            if os.path.exists(aggr_save_pth) and os.path.exists(video_save_pth) and os.path.exists(metadata_save_pth) and os.path.exists(tot_aggr_save_pth) and os.path.exists(tot_video_save_pth) and os.path.exists(tot_metadata_save_pth) and visualize and skip_done:
                if verbose:
                    print("\tContinueing since (aggregation) ALREADY DONE!!", tot_aggr_save_pth)
                continue

            ## aggregate for 'single prompt'
            # declare 'aggregator'
            aggregator = Aggregator(core=deepcopy(aggr_core))

            ## register inputs for aggregation (per prompt)
            desc = f"Aggregating for [Category '{category}' / Prompt '{prompt}' / Aggregation Setting '{aggr_settings['NAME']}' / Filter Setting '{filter_settings['filter_setting_comb']}']"
            for augid in tqdm(augid_per_cp[category][cp], desc=desc):
                # augprompt
                augprompt = augid.split(":")[0]
                # image id
                _id = augid.split(":")[1]
                    
                # input file paths
                fps = dict(
                    cam_smpl_pth = f"{filter_settings['cam_dir']}/{ldm_model_key}/{category}/{prompt}/{augprompt}:{_id}.pickle",
                    human_smpl_pth = f"{filter_settings['human_dir']}/{ldm_model_key}/{category}/{prompt}/{augprompt}/{_id}.pickle"
                )
            
                # category id
                category_id = load_category_id(category)

                # aggregation-inputs
                aggr_input_list = load_aggr_input(
                    cam_world_key=cam_world_key,
                    category_id=category_id,
                    fps=fps,
                    zerobeta=True
                )

                # iterate for all 'aggr_input'
                for _aggr_input in aggr_input_list:
                    # '_aggr_input' (cam_R, cam_t, cam_K, mask, pose, betas) -> 'aggr_input' (cam_R, cam_t, cam_K, mask, pose, betas)
                    aggr_input = aggregator.preprocess_input(
                        cam_world_key=cam_world_key,
                        **_aggr_input
                    )
                    # register inputs to 'tot_aggregator'
                    tot_aggregator.register_cam_pose(
                        cam_world_key=cam_world_key,
                        add_human_for_vis=aggr_core.GLOBAL.add_human_for_vis, # we do not visualize posed-human with zero-human
                        **aggr_input
                    )
                    # register inputs to 'aggregator'
                    aggregator.register_cam_pose(
                        cam_world_key=cam_world_key,
                        add_human_for_vis=aggr_core.GLOBAL.add_human_for_vis, # we do not visualize posed-human with zero-human
                        **aggr_input
                    )

                    # symmetric augmentation (if on)
                    if aggr_core.GLOBAL.symmetric_augmentation:
                        # apply symmetric augmentation
                        aggr_input = aggregator.convert_left_to_right(
                            cam_world_key=cam_world_key,
                            **aggr_input
                        )
                        # register augmented inputs to 'tot_aggergator'
                        tot_aggregator.register_cam_pose(
                            cam_world_key=cam_world_key,
                            add_human_for_vis=aggr_core.GLOBAL.add_human_for_vis, # we do not visualize posed-human with zero-human
                            **aggr_input
                        )
                        # register augmented inputs to 'aggregator'
                        aggregator.register_cam_pose(
                            cam_world_key=cam_world_key,
                            add_human_for_vis=aggr_core.GLOBAL.add_human_for_vis, # we do not visualize posed-human with zero-human
                            **aggr_input
                        )

            ## aggregate all registered inputs (for single prompt)
            # aggregate
            aggregator.aggregate_all(
                camera_sampling=aggr_core[cam_world_key].camera_sampling,
                verbose=verbose
            )
            # save results
            if not skip_single_prompt:
                # export 'aggregator' results
                aggregator.export(aggr_save_pth)
                # save video
                if visualize: aggregator.export_video(save_pth=video_save_pth, prompt=prompt)
                # save metadata
                metadata = extract_nonvector({'filter_settings': filter_settings, 'aggr_core': aggr_core})
                with open(metadata_save_pth, 'w') as f: json.dump(metadata, f, indent=1)


        ## skip 'total-aggr' if already done (not visualize/skip-done/skip-single-prompt & aggregation/metadata exists)
        if os.path.exists(tot_aggr_save_pth) and os.path.exists(tot_metadata_save_pth) and (not visualize) and skip_done:
            if verbose:
                print("\tContinueing since (total-aggregation) ALREADY DONE!!", tot_aggr_save_pth)
            continue
        # skip 'total-aggr' if already done (visualize/skip-done/skip-single-prompt & aggregation/video/metadata exists)
        if os.path.exists(tot_aggr_save_pth) and os.path.exists(tot_video_save_pth) and os.path.exists(tot_metadata_save_pth) and visualize and skip_done:
            if verbose:
                print("\tContinueing since (total-aggregation) ALREADY DONE!!", tot_aggr_save_pth)
            continue


        ## aggregate all registered inputs (for all prompts)
        # aggregate
        tot_aggregator.aggregate_all(
            camera_sampling=aggr_core[cam_world_key].camera_sampling,
            verbose=verbose,
        )
        # save results
        tot_aggregator.export(tot_aggr_save_pth)
        # save video
        if visualize: tot_aggregator.export_video(save_pth=tot_video_save_pth, prompt="all prompts")
        # save metadata
        metadata = extract_nonvector({'filter_settings': filter_settings, 'aggr_core': aggr_core})
        with open(tot_metadata_save_pth, 'w') as f: json.dump(metadata, f, indent=1)


def run_aggregation(cps, args):
    ## aggregate for all 'args.aggr_setting_names'
    for aggr_setting_name in args.aggr_setting_names:
        # load aggregation setting
        aggr_settings = QUANT_AGGR_SETTINGS[aggr_setting_name] if args.eval_mode == "quant" else QUAL_AGGR_SETTINGS[aggr_setting_name]
        assert aggr_settings['NAME'] == aggr_setting_name, f"Mismatch in name for '{aggr_setting_name}'"

        # retrieve filter-settings (for using images)
        for filter_setting_comb in args.filter_setting_nums:
            # print setting informations
            if args.verbose: print(f"Aggregating 'aggr_settings':{aggr_setting_name} / 'filter_setting_comb': {filter_setting_comb}")

            # filter_setting_comb as list of filter_setting_num
            filter_setting_comb = [int(i) for i in filter_setting_comb.split(",")]

            # load 'filter_settings' from multiple 'settings:00x.txt' file (while checking consistency)
            filter_settings = load_filter_settings_combination(
                args=args, 
                filter_setting_comb=filter_setting_comb,
                args_to_ignore = [
                    'categories',
                    'prompts',
                    'ldm_model_key',
                    'total_aggr_name',
                    'save_dir',
                    'aggr_setting_names',
                    'filter_setting_nums',
                    'visualize',
                    'skip_single_prompt',
                    'skip_done',
                    'verbose',
                ]
            )

            # aggregate images and save 3d results
            aggregate(
                cps=cps,
                ldm_model_key=args.ldm_model_key,
                save_dir=args.save_dir,
                cam_world_key=args.cam_world_key,
                total_aggr_name=args.total_aggr_name,
                aggr_settings=aggr_settings,
                filter_settings=filter_settings,
                visualize=args.visualize,
                skip_single_prompt=args.skip_single_prompt,
                skip_done=args.skip_done,
                verbose=args.verbose,
            )
            

if __name__ == '__main__':
    """
    [Aggregating Images from Multiple Filter-Setting]
    NOTE To enable combinations:
        Write like   "...   --filter_settings '7,8' '9'"
            => Aggregates Results from Filter Settings [7,8] 
            => Aggregates Results from Filter Settings [9]
            => [7,8] and [9] are seperate experiments
        Write like   "...   --filter_settings '7' '8' '9'"
            => Aggregates Results from Filter Settings [7]
            => Aggregates Results from Filter Settings [8]
            => Aggregates Results from Filter Settings [9]
            => [7] and [8] and [9] are seperate experiments


    [Aggregation Settings]
    NOTE Provide Aggregation Setting Names
        Write like "... --aggr_settings 'quant:full'"
            => 'eval_mode' must be set as 'quant'
            => Aggregation settings are set as 'quant:full'
            => Results are Saved Under 'quant:full' directory
            
        Write like "... --aggr_settings 'qual:002'"
            => 'eval_mode' must be set as 'qual'
            => Aggregation settings are set as 'qual:002'
            => Results are Saved Under 'qual:002' directory
    """
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--ldm_model_key", type=str, default=DEFAULT_LDM_MODEL_KEY)
    
    parser.add_argument("--filter_dir", type=str, default=DEFAULT_FILTERING_DIR)
    parser.add_argument("--cam_dir", type=str, default=DEFAULT_PERSPECTIE_CAMERA_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_AGGREGATION_DIR)
    parser.add_argument("--total_aggr_name", type=str, default=DEFAULT_TOTAL_AGGREGATION_NAME)

    parser.add_argument("--eval_mode", nargs="?", choices=["qual", "quant"])
    parser.add_argument("--cam_world_key", default='SMPL')
    
    parser.add_argument("--aggr_setting_names", nargs="+", type=str)
    parser.add_argument("--filter_setting_nums", nargs="+", type=str, default=[str(DEFAULT_FILTER_SETTING_NUM)]) # you can write as "--filter_settings '7,8' '9'"

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--skip_single_prompt", action="store_true")
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

    # pose-canonicalized aggregation
    run_aggregation(cps=cps, args=args)