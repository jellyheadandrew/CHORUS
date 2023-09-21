import os
from glob import glob
import argparse
from tqdm import tqdm

import pickle5 as pickle
from copy import deepcopy
from easydict import EasyDict

import torch
import cv2

import imports.frankmocap.mocap_utils.demo_utils as demo_utils

from constants.ldm import DEFAULT_LDM_MODEL_KEY
from constants.frankmocap import DEFAULT_EXCONF_FRANKMOCAP, DEFAULT_MINOVERLAP_FRANKMOCAP
from constants.metadata import DEFAULT_IMAGE_DIR, DEFAULT_SEGMENTATION_DIR, DEFAULT_HUMAN_DIR, DEFAULT_SEED

from utils.prepare_prompts import prepare_cps_from_dirs
from utils.prepare_frankmocap import prepare_frankmocap_regressor, prepare_frankmocap_visualizer
from utils.postprocess import process_bbox_mask, process_mocap_predictions, process_remove_none, process_remove_overlap, process_segmentation, bbox_xy_to_wh
from utils.reproducibility import seed_everything



def extract_smpl(hparams, img_pth, seg_pth, bodymocap, visualizer):
    # load image
    image_bgr=cv2.imread(img_pth)

    # load segmentation results
    with open(seg_pth, "rb") as handle:
        instances_orig=pickle.load(handle)
    instances_orig=instances_orig.to("cuda")


    # post-process all objects (excluding humans) before post-processing for humans 
    if hparams.remocc:
        # remove excessive occlusions for same categories
        instances = process_segmentation(
            deepcopy(instances_orig),
            minoverlap=hparams.minoverlap, 
            exconf=hparams.exconf,
            verbose=hparams.verbose,
        )
        # print progress
        if hparams.verbose:
            print("[Log] Removed excessive occlusions from image segmentation results.")
    else:
        instances = deepcopy(instances_orig)

    # find human predictions (2D)
    is_person = (instances.pred_classes == 0)
    human_indices = torch.tensor(list(range(len(instances.pred_classes))), device=is_person.device)[is_person]
    bboxes_person = (instances[is_person].pred_boxes.tensor.cpu().numpy()) # bbox type: xyxy
    masks_person = instances[is_person].pred_masks
    confidence_person = instances[is_person].scores.cpu().numpy() # ndarray of shape [num_human,]

    # if no humans were detected during segmentation
    if bboxes_person.shape[0] == 0:
        if hparams.verbose:
            print("No human detected --> ",img_pth)

        # save image with "no human" message
        image_bgr = cv2.putText(
            image_bgr, 
            text="NO HUMAN DETECTED", 
            org = (image_bgr.shape[0]-150, image_bgr.shape[0]-20),
            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            fontScale=0.5,
            color=(0,0,255))
        return None, image_bgr

    # if humans were detected during segmentation, post-process bbox adaptable for human-prediction model
    body_bbox_list = bbox_xy_to_wh(bboxes_person)  # numpy ndarray, shape: [num_human, 4], xyxy -> xywh
    confidence_list = confidence_person.tolist()

    # calculate the overlapping bboxes and remove if too much is overlapping
    if hparams.remocc:
        keepidx=process_remove_overlap(
            bbox_list=body_bbox_list.tolist(), 
            confidence_list=confidence_list, 
            minoverlap=hparams.minoverlap, 
            exconf=hparams.exconf
        )
    else:
        keepidx=list(range(len(body_bbox_list)))
    
    # if there are no human remaining, return image only
    if len(keepidx) == 0:
        if hparams.verbose:
            print("After removing overlapping, no human exists.")
        return None, image_bgr


    # regress body pose
    mocap_output_list = bodymocap.regress(image_bgr, body_bbox_list[keepidx])
    
    # if erroneous mocap-output exists, remove (cases include extremely small bboxes)
    if None in mocap_output_list:
        if hparams.verbose:
            print(f"Erroneous human prediction exists for {img_pth}\n--> Removing Erroneous")
    mocap_output_list, final_keepidx = process_remove_none(mocap_output_list, keepidx)
    
    # if there are no humans remaining, return image only
    if len(mocap_output_list) == 0:
        if hparams.verbose:
            print("After removing erroneous, no human exists.")
        return None, image_bgr

    # post-process remaining human bboxes & masks
    final_bboxes_person, final_masks_person = process_bbox_mask(bboxes_person, masks_person, final_keepidx)

    # extract SMPL mesh for rendering (vertices, faces in image space)
    pred_mesh_list = demo_utils.extract_mesh_from_output(mocap_output_list)

    # render SMPL mesh on image
    if visualizer is not None:
        res_img = visualizer.visualize(
            image_bgr,
            pred_mesh_list = pred_mesh_list, 
            body_bbox_list = body_bbox_list[final_keepidx])
    else:
        res_img = None

    # process mocap predictions to save (NOTE: order-changing happens inside this function)
    human_parameters = process_mocap_predictions( # boxes in human_parameters['bboxes'] --> Shape: (N,4), xyxy format
        mocap_predictions=mocap_output_list,
        bboxes=final_bboxes_person, # xyxy
        masks=final_masks_person,
        image_size=max(image_bgr.shape)
    )
    for key in human_parameters:
        human_parameters[key]=human_parameters[key].to("cpu")

    # direct outputs
    human_parameters['mocap_output_list'] = mocap_output_list
    human_parameters['body_bbox_list'] = body_bbox_list[final_keepidx] # xywh format, (N,4)
    human_parameters['body_bbox_list_xyxy'] = bboxes_person[final_keepidx] # xyxy format, (N,4)
    human_parameters['confidence_list'] = confidence_person[final_keepidx] # (N,)

    # original SMPL outputs
    human_parameters['rotmat'] = torch.stack([torch.tensor(pred_output['pred_rotmat'][0]) for pred_output in mocap_output_list], dim=0)
    human_parameters['betas'] = torch.stack([torch.tensor(pred_output['pred_betas'][0]) for pred_output in mocap_output_list], dim=0)
    
    # SMPL joints (not permutated)
    human_parameters['joint_vertices_smpl'] = torch.stack([torch.tensor(pred_output['pred_joints_smpl'][0]) for pred_output in mocap_output_list], dim=0)

    # re-order (left-bbox to right-bbox)
    indices = human_parameters.pop('indices')
    for key in ['rotmat', 'betas', 'joint_vertices_smpl']:
        human_parameters[key] = human_parameters[key][indices]
        
    # meshes & bboxes
    human_parameters['pred_mesh_list'] = pred_mesh_list
    human_parameters['body_bbox_list'] = body_bbox_list[final_keepidx]

    # indices of removed humans in original instances
    removed_human_idx=list(range(len(human_indices)))
    for survived_idx in final_keepidx:
        removed_human_idx.remove(survived_idx)

    # auxiliary information
    human_parameters['auxiliary'] = dict()
    human_parameters['auxiliary']['instances_including_object_original'] = instances_orig.to("cpu") # original instances
    human_parameters['auxiliary']['instances_including_object'] = instances.to("cpu") # set of instances where all humans and post-processed objects exist
    human_parameters['auxiliary']['total_human_indices'] = human_indices # human indices used to query 'instances including object'
    human_parameters['auxiliary']['removed_human_indices']=human_indices[removed_human_idx].detach().cpu().numpy() # indices of removed humans in original instances
    human_parameters['auxiliary']['not_removed_human_indices']=human_indices[final_keepidx].detach().cpu().numpy() # indices of remaining humans in original instances
    
    return human_parameters, res_img


def human_prediction(
        cps,
        ldm_model_key,
        image_dir,
        seg_dir,
        save_dir,
        exconf,
        minoverlap,
        remocc,
        use_visualizer,
        skip_done,
        verbose,
    ):
    
    # hyperparameters
    hparams = EasyDict(dict(exconf=exconf,minoverlap=minoverlap,remocc=remocc,verbose=verbose))

    # prepare bodymocap model
    bodymocap = prepare_frankmocap_regressor()
    
    # prepare visualizer to render human    
    visualizer = prepare_frankmocap_visualizer() if use_visualizer else None

    # clear screen if not verbose
    if not verbose: os.system("clear")
    
    # run human prediction for all (category, prompt) pairs
    for category, prompt in cps:
        # all image path
        img_pths = sorted(list(glob(f"./{image_dir}/{ldm_model_key}/{category}/{prompt}/*/*.png")))

        # run for all images
        for img_pth in tqdm(img_pths, desc=f"Predicting Humans for [Category '{category}' / Prompt '{prompt}']"):
            # print
            if verbose:
                print("Processing: ", img_pth)
            
            # augprompt
            augprompt = img_pth.split("/")[-2]
            # image id
            _id, _ext = img_pth.split("/")[-1].split(".")            
            # result-save-directory
            result_save_dir = f"./{save_dir}/{ldm_model_key}/{category}/{prompt}/{augprompt}"
            os.makedirs(result_save_dir, exist_ok=True)
            # result-save paths
            img_save_pth = os.path.join(result_save_dir, f"{_id}.png")
            human_save_pth = os.path.join(result_save_dir, f"{_id}.pickle")

            # skip if already done
            if use_visualizer and os.path.exists(img_save_pth) and os.path.exists(human_save_pth) and skip_done:
                if verbose:
                    print("\tContinueing since ALREADY DONE!!", human_save_pth)
                continue
            if not use_visualizer and os.path.exists(human_save_pth) and skip_done:
                if verbose:
                    print("\tContinueing since ALREADY DONE!!", human_save_pth)
                continue

            # load segmentation results
            seg_pth = img_pth.replace(".png", ".pickle").replace(image_dir, seg_dir)
            if not os.path.exists(seg_pth):
                if verbose:
                    print("\tContinueing because there is no segmentation result: ", seg_pth)
                continue

            # extract human (as SMPL) with post-processing / (optional) render SMPL human on image
            human_params, rendered_img = extract_smpl(
                hparams=hparams,
                img_pth=img_pth, 
                seg_pth=seg_pth, 
                bodymocap=bodymocap,
                visualizer=visualizer
            )
            # error cases
            assert not (human_params is None and rendered_img is None), "error case during human prediction"
            
            # if no human was detected
            if human_params is None:
                cv2.imwrite(img_save_pth, rendered_img)
                with open(human_save_pth, "wb") as handle:
                    pickle.dump("NO HUMANS", handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
            # if human was detected
            else:
                # save human-rendered image if it exists 
                if use_visualizer:
                    imglength=rendered_img.shape[0]
                    cv2.imwrite(img_save_pth, rendered_img[:,imglength:])

                # save SMPL results
                with open(human_save_pth, "wb") as handle:
                    pickle.dump(human_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--ldm_model_key", type=str, default=DEFAULT_LDM_MODEL_KEY)

    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--seg_dir", type=str, default=DEFAULT_SEGMENTATION_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_HUMAN_DIR)

    parser.add_argument("--exconf", type=float, default=DEFAULT_EXCONF_FRANKMOCAP) # exception confidence (maximum)
    parser.add_argument("--minoverlap", type=float, default=DEFAULT_MINOVERLAP_FRANKMOCAP) # minimum overlap ratio
    parser.add_argument("--disable_remocc", action="store_true")
    parser.add_argument("--use_visualizer", action="store_true")

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

    # run human prediction
    human_prediction(
        cps=cps,
        ldm_model_key=args.ldm_model_key,
        image_dir=args.image_dir,
        seg_dir=args.seg_dir,
        save_dir=args.save_dir,
        exconf=args.exconf,
        minoverlap=args.minoverlap,
        remocc=not args.disable_remocc,
        use_visualizer=args.use_visualizer,
        skip_done=args.skip_done,
        verbose=args.verbose,
    )