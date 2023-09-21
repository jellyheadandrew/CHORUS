import os
from glob import glob
import shutil

import argparse
from tqdm import tqdm

import random
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt

from constants.ldm import DEFAULT_LDM_MODEL_KEY
from constants.filtering import KEYPOINT_FILTERS, DEFAULT_KEYPOINT_FILTER_NUM, DEFAULT_FILTER_MULTI_PERSON_OBJECT_METHODS, DEFAULT_BBOX_OVERLAP_THRES, DEFAULT_KEYPOINT_CONFIDENCE_THRES
from constants.metadata import DEFAULT_IMAGE_DIR, DEFAULT_SEGMENTATION_DIR, DEFAULT_HUMAN_DIR, DEFAULT_KEYPOINT_DIR, DEFAULT_FILTERING_DIR, DEFAULT_SEED

from utils.settings import create_filter_settings, find_filter_setting_num
from utils.prepare_prompts import prepare_cps_from_dirs
from utils.filtering import bbox_iou
from utils.reproducibility import seed_everything
from utils.misc import load_category_id




def apply_filters(create_settings_only, cps, **kwargs):

    # retrieve settings
    settings_str, settings = create_filter_settings(**kwargs)
    
    # used for counting filtered samples in category
    curr_category = None

    # find setting number
    setting_num = find_filter_setting_num(settings_str, settings)

    # if create settings only
    if create_settings_only:
        if settings['verbose']:
            print("\tRunning 'create_settings_only'...")
        return

    # clear screen if not verbose
    if not settings['verbose']: os.system("clear")

    # run filtering for all (category, prompt) pairs
    for category, prompt in sorted(cps):
        # number of samples retained (for given category)
        if curr_category != category:
            num_retained_in_category = 0
            curr_category = category
    
        # all image path
        img_pths = sorted(list(glob(f"./{settings['image_dir']}/{settings['ldm_model_key']}/{category}/{prompt}/*/*.png")))
        num_retained_in_prompt = 0 # counter for useful samples (not filtered)

        # iterate through image path
        for img_pth in tqdm(img_pths, desc=f"Filtering Dataset for [Category '{category}' / Prompt '{prompt}' / Setting: {setting_num:03}]"):
            # print
            if settings['verbose']:
                print("Processing: ", img_pth)

            # augprompt
            augprompt = img_pth.split("/")[-2]
            # image id
            _id, _ext = img_pth.split("/")[-1].split(".")
            # result-save directory
            result_save_dir = f"./{settings['save_dir']}/{settings['ldm_model_key']}/{category}/settings:{setting_num:03}/{prompt}"
            os.makedirs(result_save_dir, exist_ok=True)
            os.makedirs(f"{result_save_dir}/FILTERED", exist_ok=True)
            # result-save paths
            retained_save_pth =f"{result_save_dir}/{augprompt}:{_id}.png"
            filtered_save_pth=f"{result_save_dir}/FILTERED/{augprompt}:{_id}"

            # skip if already done
            if os.path.exists(retained_save_pth) and settings['skip_done']:
                num_retained_in_prompt += 1
                num_retained_in_category += 1
                if settings['verbose']:
                    print(f"\tContinueing since ALREADY DONE (retained)!!", retained_save_pth) 
                continue
            if os.path.exists(filtered_save_pth) and settings['skip_done']:
                if settings['verbose']:
                    print(f"\tContinueing since ALREADY DONE (filtered)!!", filtered_save_pth) 
                continue

            ## loading human information & filtering
            # load human params
            human_pth = f"./{settings['human_dir']}/{settings['ldm_model_key']}/{category}/{prompt}/{augprompt}/{_id}.pickle"
            with open(human_pth, "rb") as handle: human_parameters = pickle.load(handle)

            # if there is no human, filtering is applied
            if human_parameters == "NO HUMANS":
                with open(filtered_save_pth, "w") as wf: wf.write("NO HUMAN")
                if settings['verbose']: print(f"\tFiltering {filtered_save_pth} since there doesn't exist any humans...")
                continue

            # if there is more than 1 human, filtering is applied (if skip_multi_person_image)
            num_humans = len(human_parameters['cams'])
            if num_humans >= 2 and settings['skip_multi_person_image']:
                with open(filtered_save_pth, "w") as wf: wf.write("MORE THAN 2 HUMAN")
                if settings['verbose']: print(f"\tFiltering {filtered_save_pth} since there exists more than 2 humans for single camera...")
                continue
        
            ## loading object information & filtering
            # category id
            category_id = load_category_id(category)
            
            #  load object mask & bbox (xyxy)
            instances = human_parameters['auxiliary']['instances_including_object']
            is_class = instances.pred_classes == category_id
            object_masks = instances.pred_masks[is_class]
            object_masks = object_masks.detach().cpu().numpy() # [N,img_size,img_size]
            object_bboxes_xyxy = instances[is_class].pred_boxes.tensor.detach().cpu().numpy() # [N,4]
            assert object_masks.shape[0] == object_bboxes_xyxy.shape[0]

            # if there are no objects in image, filtering is applied
            if object_masks.shape[0] == 0:
                with open(filtered_save_pth, "w") as wf:
                    wf.write("NO TARGET OBJECT")
                if settings['verbose']:
                    print(f"\tFiltering {filtered_save_pth} since there are no corresponding object mask (for given target category)")
                continue
            
            # if there are multiple objects in image, filtering is applied
            elif object_masks.shape[0] >= 2 and settings['skip_multi_object_image']:
                with open(filtered_save_pth, "w") as wf:
                    wf.write("MORE THAN 2 TARGET OBJECTS")
                if settings['verbose']:
                    print(f"\tFiltering {filtered_save_pth} since there are more than 2 object masks (for given target category)")
                continue

            ## checking object-bbox & human-bbox overlap and filtering
            # find overlapping bbox (criteria: intersection over smaller bbox area)
            overlap_bbox = []
            
            # iterate for all humans
            to_filter = False
            for human_idx in range(num_humans):
                human_bbox_xyxy = human_parameters['body_bbox_list_xyxy'][human_idx]
                for target_obj_idx in range(object_bboxes_xyxy.shape[0]):
                    target_obj_bbox_xyxy = object_bboxes_xyxy[target_obj_idx]
                    iou, human_bbox_area, obj_bbox_area, intersection_area = bbox_iou(human_bbox_xyxy, target_obj_bbox_xyxy, return_areas_too=True)
                    if intersection_area / obj_bbox_area > settings['bbox_overlap_thres']:
                        overlap_bbox.append(target_obj_idx)
            
                # if there are no restrictions for number of object-bboxes that overlap with human-bbox, use all object-bboxes
                if len(settings['allowed_bbox_num_per_human']) == 0:
                    object_mask = np.any(object_masks, axis=0)

                # if number of object-bboxes that overlap with human-bbox is not allowed, filtering is applied
                elif len(overlap_bbox) not in settings['allowed_bbox_num_per_human']:
                    to_filter = True

                # if number of object-bboxes that overlap with human-bbox is allowed, merge the overlapping object-bboxes (not all)
                else:
                    object_mask = np.any(object_masks[overlap_bbox], axis=0)

            # apply filtering
            if to_filter:
                with open(filtered_save_pth, "w") as wf:
                    wf.write("MORE THAN 2 VALID BBOX PER PERSON")
                if settings['verbose']:
                    print(f"\tFiltering {filtered_save_pth} since number of object-bbox that overlaps with human-bbox: {len(overlap_bbox)} is not allowed in: {settings['allowed_bbox_num_per_human']}")
                continue

            ## checking human 2D keypoints and filtering
            # retrieve human parameters (for checking consistency of results)
            mocap_output_list = human_parameters["mocap_output_list"]
                    
            ## checking coco-keypoints and filtering
            # load coco-keypoint prediction results (if we do filter)
            if len(settings['keypoints_that_must_exist']) > 0 or settings['keypoints_that_must_not_exist'] > 0:
                keypoint_coco_pth = f"./{settings['keypoint_dir']}/{settings['ldm_model_key']}/{category}/{prompt}/{augprompt}/{_id}.pickle"
                with open(keypoint_coco_pth, "rb") as handle:
                    keypoints_per_human = pickle.load(handle)
                assert len(keypoints_per_human) == len(mocap_output_list) # check the consistency via number of humans

            # iterate for all humans
            to_filter = False
            for human_idx in range(num_humans):
                # if all of elements in the "subset of coco-keypoints that 'one of them must exist'" has lower confidence than threshold, filtering is applied
                for query_idx_tuple in settings['keypoints_that_must_exist']:
                    # note that at least one joint in 'query_idx_tuple' must have higher confidence than threshold
                    all_must_exist_kp_under_thres = True
                    for query_idx in query_idx_tuple:
                        x_img, y_img, confidence = keypoints_per_human[human_idx]['keypoints'][query_idx][:].tolist()
                        if confidence >= settings['keypoint_confidence_threshold']:
                            all_must_exist_kp_under_thres = False
                            break
                    if all_must_exist_kp_under_thres:
                        to_filter = True
                        break

                # if all of elements in the "subset of coco-keypoints that 'one of them must-not exist'" has higher confidence than threshold, filtering is applied
                for query_idx_tuple in settings['keypoints_that_must_not_exist']:
                    # note that at least one joint in 'query_idx_tuple' must have higher confidence than threshold
                    all_mustnot_exist_kp_over_thres = True
                    for query_idx in query_idx_tuple:
                        x_img, y_img, confidence = keypoints_per_human[human_idx]['keypoints'][query_idx][:].tolist()
                        if confidence < settings['keypoint_confidence_threshold']:
                            all_mustnot_exist_kp_over_thres = False
                            break
                    if all_mustnot_exist_kp_over_thres:
                        to_filter = True
                        break

            # apply filtering
            if to_filter:
                with open(filtered_save_pth, "w") as wf:
                    wf.write("FILTERED VIA COCO KEYPOINTS")
                if settings['verbose']:
                    print(f"\tFiltering {filtered_save_pth} via 2D COCO-keypoint!!!")
                continue
                                
            ## after all these steps of filtering, sample is retained. add to the number for tracking
            num_retained_in_prompt += 1
            num_retained_in_category += 1

            ## save as 'png' format if not filtered
            with open(retained_save_pth, "w") as wf:
                wf.write(f"useful_in_category:{num_retained_in_category:06}/useful_in_prompt:{num_retained_in_prompt:06}")


def filtering(cps, args):
    # prepare keypoint filters
    if args.keypoint_filter_nums is None:
        if args.verbose:
            print("'keypoint_filter_nums' not specified: Running for all keypoint filters!!!")
        args.keypoint_filter_nums = sorted(list(KEYPOINT_FILTERS.keys()))
    else:
        pass

    ## creating settings first
    keypoint_filter_nums = sorted(list(KEYPOINT_FILTERS.keys()))
    for kpfilter_idx in keypoint_filter_nums:
        # only save settings
        apply_filters(
            create_settings_only=True,
            cps=None,
            ldm_model_key=args.ldm_model_key,
            image_dir=args.image_dir,
            seg_dir=args.seg_dir,
            human_dir=args.human_dir,
            save_dir=args.save_dir,
            keypoint_dir=args.keypoint_dir,
            use_template=args.use_template,
            keypoints_which = KEYPOINT_FILTERS[kpfilter_idx],
            filter_multi_person_object_methods = args.filter_multi_person_object_methods,
            bbox_overlap_thres=args.bbox_overlap_thres,
            keypoint_confidence_threshold=args.keypoint_confidence_threshold,
            skip_done=args.skip_done,
            verbose=args.verbose,
        )

    ## do filtering
    keypoint_filter_nums = args.keypoint_filter_nums
    for kpfilter_idx in keypoint_filter_nums:
        # Run filtering
        apply_filters(
            create_settings_only=False,
            cps=cps,
            ldm_model_key=args.ldm_model_key,
            image_dir=args.image_dir,
            seg_dir=args.seg_dir,
            human_dir=args.human_dir,
            save_dir=args.save_dir,
            keypoint_dir=args.keypoint_dir,
            use_template=args.use_template,
            keypoints_which = KEYPOINT_FILTERS[kpfilter_idx],
            filter_multi_person_object_methods = args.filter_multi_person_object_methods,
            bbox_overlap_thres=args.bbox_overlap_thres,
            keypoint_confidence_threshold=args.keypoint_confidence_threshold,
            skip_done=args.skip_done,
            verbose=args.verbose,
        )
            

if __name__ == "__main__":
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--ldm_model_key", type=str, default=DEFAULT_LDM_MODEL_KEY)

    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--seg_dir", type=str, default=DEFAULT_SEGMENTATION_DIR)
    parser.add_argument("--human_dir", type=str, default=DEFAULT_HUMAN_DIR)
    parser.add_argument("--keypoint_dir", type=str, default=DEFAULT_KEYPOINT_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_FILTERING_DIR)
    parser.add_argument("--use_template", action="store_true") # --> overrides 'seg_dir', 'human_dir', 'keypoint_dir'

    parser.add_argument("--keypoint_filter_nums", nargs="+", type=int, default=[DEFAULT_KEYPOINT_FILTER_NUM], choices=sorted(list(KEYPOINT_FILTERS.keys())))
    parser.add_argument("--filter_multi_person_object_methods", default=DEFAULT_FILTER_MULTI_PERSON_OBJECT_METHODS) # multi-person/object filtering
    parser.add_argument("--bbox_overlap_thres", type=float, default=DEFAULT_BBOX_OVERLAP_THRES) # bbox-overlap filtering
    parser.add_argument("--keypoint_confidence_threshold", type=float, default=DEFAULT_KEYPOINT_CONFIDENCE_THRES) # keypoint filtering confidence threshold

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

    # filtering
    filtering(cps=cps, args=args)