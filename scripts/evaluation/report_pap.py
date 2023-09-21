import os
from glob import glob
import argparse
import json
from easydict import EasyDict
from tqdm import tqdm

from constants.ldm import DEFAULT_LDM_MODEL_KEY
from constants.metadata import DEFAULT_QUANT_EVAL_DIR


def report_pap(args):
    ## placeholder for saving 'mean' evaluation results
    eval_total_results = dict()

    # evaluation-result paths
    eval_pths = sorted(list(glob(f"./{args.quant_eval_dir}/{args.ldm_model_key}/*/*.json")))
    # iterate for all quantitative-evaluation results
    for eval_pth in tqdm(eval_pths, desc="Reporting mPAP metrics..."):
        # open quantitative-evaluation results
        with open(eval_pth, "r") as rf:
            data = json.load(rf)
        category = data["category"]
        # skip if not in 'args.categories'
        if args.categories is not None:
            if category not in args.categories: continue
        
        # model_name
        model_name = data["aggr_pth"].split("/")[-1].replace(".pickle", "")

        # retrieve results        
        results = data["results"]
        mPAP = results["mPAP"]
        mPAP_strict = results["mPAP_strict"]
        mPAP_hoa = results["mPAP_hoa"]
        mPAP_strict_hoa = results["mPAP_strict_hoa"]

        # if the case of computing 'mean' mPAP, add to placeholder
        if args.categories is None and model_name not in eval_total_results.keys():
            eval_total_results[model_name] = dict()
            eval_total_results[model_name]['category_count'] = 1
            eval_total_results[model_name]['mPAP_mean'] = mPAP
            eval_total_results[model_name]['mPAP_strict_mean'] = mPAP_strict
            eval_total_results[model_name]['mPAP_hoa_mean'] = mPAP_hoa
            eval_total_results[model_name]['mPAP_strict_hoa_mean'] = mPAP_strict_hoa

        elif args.categories is None and model_name in eval_total_results.keys():
            eval_total_results[model_name]['category_count'] += 1
            eval_total_results[model_name]['mPAP_mean'] += mPAP
            eval_total_results[model_name]['mPAP_strict_mean'] += mPAP_strict
            eval_total_results[model_name]['mPAP_hoa_mean'] += mPAP_hoa
            eval_total_results[model_name]['mPAP_strict_hoa_mean'] += mPAP_strict_hoa

    # in the 'mean' mPAP evaluation case
    if args.categories is None:
        for model_name in eval_total_results.keys():
            total_count = eval_total_results[model_name]['category_count']
            mPAP_avg = eval_total_results[model_name]['mPAP_mean'] / total_count
            mPAP_strict_avg = eval_total_results[model_name]['mPAP_strict_mean'] / total_count
            mPAP_hoa_avg = eval_total_results[model_name]['mPAP_hoa_mean'] / total_count
            mPAP_strict_hoa_avg = eval_total_results[model_name]['mPAP_strict_hoa_mean'] / total_count

            print(f"\n==Aggregation Settings: {model_name}==")
            print(f"Average mPAP: {mPAP_avg * 100:.2f}")
            print(f"Average mPAP_strict: {mPAP_strict_avg * 100:.2f}")
            print(f"Average mPAP_hoa: {mPAP_hoa_avg * 100:.2f}")
            print(f"Average mPAP_strict_hoa: {mPAP_strict_hoa_avg * 100:.2f}")
            print(f"Categories used: {total_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", type=str) # set as 'None' for default
    parser.add_argument("--ldm_model_key", type=str, default=DEFAULT_LDM_MODEL_KEY)
    parser.add_argument("--quant_eval_dir", default=DEFAULT_QUANT_EVAL_DIR)
    args = parser.parse_args()
    
    report_pap(args=args)