import os
import argparse
from copy import deepcopy
from easydict import EasyDict
import pickle5 as pickle
from tqdm import tqdm
import json

import numpy as np
import torch

import cv2
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

from imports.frankmocap.bodymocap.models import SMPL

from constants.frankmocap import BODY_MOCAP_SMPL_PTH
from constants.prepare_eft import DEFAULT_EFT_MIN_BBOX_OVERLAP, DEFAULT_EFT_REPROJECTION_ERROR_THRESH, DEFAULT_EFT_FOV, DEFAULT_EFT_LR, DEFAULT_EFT_JOINT_LOSS_THRES, DEFAULT_EFT_EPOCHS
from constants.datasets import COCO_VAL_ANNOTATION_PTH, COCO_VAL_IMAGE_DIR, COCO_EFT_VAL_PTH, COCO_CLASS_NAME2ID, EXTENDED_COCO_EFT_VAL_SAVE_PTH
from constants.metadata import DEFAULT_SEED

from utils.reproducibility import seed_everything
from utils.filtering import intersection_over_smaller_bbox
from utils.transformations import rotmat_to_axis_angle_batch
from utils.frankmocap_conversion import convert_smpl_to_img
from utils.perspective_conversion import apply_wp_to_p


def extend_coco_eft_dataset(args):
    ## prepare smpl model & coco-constants
    smpl = SMPL(BODY_MOCAP_SMPL_PTH, batch_size=1, create_transl=False)
    coco=COCO(COCO_VAL_ANNOTATION_PTH)
    person_category_id = 1

    ## open COCO-EFT (val) dataset
    # load eft-dataset
    with open(COCO_EFT_VAL_PTH, "r") as rf: eft_data = json.load(rf)

    # print EFT-data version
    if args.verbose:
        print("EFT data: ver {}".format(eft_data['ver']))

    # original eft data
    orig_eft_data = eft_data['data']

    # default categories: all coco-categories
    if args.categories is None: args.categories = list(COCO_CLASS_NAME2ID.keys())

    # placeholder to save 'extended data'
    eft_data_save = dict()
    stats_per_category = dict()

    # clear screen if not verbose
    if not args.verbose: os.system("clear")

    ## iterate for all 'categories'
    for category in args.categories:
        # skip 'person' category
        if category == 'person':
            continue

        # get category_id (note: not contiguous-id)
        category_id = coco.getCatIds(catNms=[category])[0]

        # count number-of-processed data
        num_filtered = 0
        desc = f"Extending COCO-EFT Dataset for [Category: '{category}'] --> Num-Filtered: ({num_filtered})"
        
        ## iterate for all data in 'eft-dataset' (note: each data corresponds to 'smpl-body annotation')
        pbar = tqdm(enumerate(sorted(orig_eft_data, key = lambda x: x['imageName'])), desc=desc)
        for eft_idx, orig_data in pbar:
            # if this data is already processed (and saved) for any previous 'category', we append info from current 'category'
            if eft_idx in eft_data_save.keys():
                # print existence
                if args.verbose: print(f"eft_idx:{eft_idx} already exists in 'eft_data_save'")
                # copy data
                data = deepcopy(eft_data_save[eft_idx])
                is_overridden = True
                if args.verbose: print(f"while {data['target_category_list']} exists, we are now Updating for {category}")
            else:
                is_overridden = False
                data = deepcopy(orig_data)

            ## ids
            # load image
            img_name = data['imageName']
            img_pth = f'{COCO_VAL_IMAGE_DIR}/{img_name}'
            img_id = int(img_name.replace("COCO_val2014_", "").replace(".jpg", ""))
            img = cv2.cvtColor(cv2.imread(img_pth), code=cv2.COLOR_BGR2RGB)

            # coco-annotation-ids for category 'person'
            person_annIds = coco.getAnnIds(imgIds=img_id, catIds=[person_category_id], iscrowd=None)
            
            # coco-annotation-ids for category
            object_annIds = coco.getAnnIds(imgIds=img_id, catIds=[category_id], iscrowd=None)

            ## filter 'multi-person' / 'multi-object' images
            # skip if multiple objects exist in the image
            if len(object_annIds) != 1:
                if args.verbose:
                    print(f"Continue {category} for this image since multiple objects or no object")
                continue

            # skip if multiple people exist in the image
            if len(person_annIds) != 1:
                if args.verbose:
                    print("Continue since multiple person or no person")
                continue

            ## load object-bboxes & object-masks
            # load coco-annotations from 'coco-annotation-ids'
            person_anns = coco.loadAnns(person_annIds)
            object_anns = coco.loadAnns(object_annIds)

            # declare placeholder for saving 'bbox', 'mask' and precomputed 'bbox-bbox_overlap'
            if is_overridden:
                assert 'bbox' in data.keys()
                assert category not in data['bbox'].keys()
            else:
                assert 'bbox' not in data.keys()
                data['bbox'] = {}
            if is_overridden:
                assert 'mask' in data.keys()
                assert category not in data['mask'].keys()
            else:
                assert 'mask' not in data.keys()
                data['mask'] = {}
            if is_overridden:
                assert 'bbox_overlap_per_category' in data.keys()
                assert category not in data['bbox_overlap_per_category']
            else:
                assert 'bbox_overlap_per_category' not in data.keys()
                data['bbox_overlap_per_category'] = {}

            # mask and bbox for category 'person'
            person_mask = coco.annToMask(person_anns[0])
            [x,y,w,h] = person_anns[0]['bbox']
            person_bbox_xyxy = [x,y,x+w,y+h]
                        
            # mask and bbox for category 'object'
            object_mask = coco.annToMask(object_anns[0])
            [x,y,w,h] = object_anns[0]['bbox']
            object_bbox_xyxy = [x,y,x+w,y+h]

            ## filter by 'bbox-bbox_overlap'
            # compute 'bbox-overlap', and save (even if filtered)
            bbox_overlap = intersection_over_smaller_bbox(person_bbox_xyxy, object_bbox_xyxy)
            data['bbox_overlap_per_category'][category] = bbox_overlap

            # filter images with small 'bbox-overlap'
            if bbox_overlap < args.min_bbox_overlap:
                if args.verbose:
                    print("Continue since bbox bbox_overlap is too small")
                continue

            ## fill in 'target-category', 'mask' and 'bbox'
            # fill in 'target category list'
            if is_overridden:
                assert 'target_category_list' in data.keys()
                assert category not in data.keys()
                data['target_category_list'].append(category)
            else:
                assert 'target_category_list' not in data.keys()
                data['target_category_list'] = [category]
            
            # fill in 'mask' and 'bbox' of person/object
            if is_overridden:
                assert 'bbox' in data.keys()
                assert 'mask' in data.keys()
                assert 'person' in data['bbox'].keys()
                assert 'person' in data['mask'].keys()
                assert category not in data['bbox'].keys()
                assert category not in data['mask'].keys()
                data['bbox'][category] = object_bbox_xyxy
                data['mask'][category] = (object_mask > 0).astype(np.uint8)                
            else:
                data['bbox']['person'] = person_bbox_xyxy
                data['mask']['person'] = (person_mask > 0).astype(np.uint8)
                data['bbox'][category] = object_bbox_xyxy
                data['mask'][category] = (object_mask > 0).astype(np.uint8)
                

            ## compute & fill in 'perspective camera'
            if is_overridden:
                assert 'pred_aa' in data.keys()
                assert 'pred_betas' in data.keys()
                assert 'persp_cam_list' in data.keys()
                assert 'smpl_joints_2d' in data.keys()
                persp_cam_smpl_list = data['persp_cam_list']
                pred_aa = data['pred_aa']
                pred_betas = data['pred_betas']
                smpl_joints_2d = data['smpl_joints_2d']
            else:
                assert 'pred_aa' not in data.keys()
                assert 'pred_betas' not in data.keys()
                assert 'persp_cam_list' not in data.keys()
                assert 'smpl_joints_2d' not in data.keys()

                ## prepare data required
                # bbox scale & center
                bbox_scale = data['bbox_scale']
                bbox_center = data['bbox_center']
                # weak-perspective camera
                pred_camera = np.array(data['parm_cam'])
                cam_param_scale = pred_camera[0]
                cam_param_trans = pred_camera[1:]
                # smpl pose & betas -> set betas as zero
                pred_betas = np.reshape(np.array( data['parm_shape'], dtype=np.float32),(1,10))            #(1,10)
                pred_betas = torch.from_numpy(pred_betas)
                pred_pose_rotmat = np.reshape(np.array( data['parm_pose'], dtype=np.float32), (1,24,3,3))  #(1,24,3,3)
                pred_pose_rotmat = torch.from_numpy(pred_pose_rotmat)
                
                axis, theta = rotmat_to_axis_angle_batch(pred_pose_rotmat[0]) # (24,3), (24,)
                pred_aa = axis * theta[:,None]
                pred_aa = pred_aa.reshape(1,72)

                ## retrieve smpl-mesh
                # infer smpl mesh
                smpl_output = smpl( 
                    betas=pred_betas, 
                    body_pose=pred_aa[:,3:],
                    global_orient=pred_aa[:,:3], 
                    pose2rot=True
                )
                smpl_output_zerorot_zerobeta = smpl( 
                    betas=torch.zeros_like(pred_betas), 
                    body_pose=pred_aa[:,3:],
                    global_orient=torch.zeros_like(pred_aa[:,:3]), 
                    pose2rot=True
                )
                
                # smpl vertices & joints in '3d-smpl-world'
                smpl_joints_3d = smpl_output.joints.detach().cpu().numpy()[0]

                # conversion of joints in '3d-smpl-world' to '2d-image-world'
                smpl_joints_2d = convert_smpl_to_img(
                    smpl_3d=smpl_joints_3d, 
                    img=img, 
                    bbox_center=bbox_center, 
                    bbox_scale=bbox_scale, 
                    cam_param_scale=cam_param_scale, 
                    cam_param_trans=cam_param_trans
                )

                # create a 'mocap_output_list' to use as input for 'weak-perspective to perspective' conversion
                mocap_output_list = [dict(
                    pred_joints_img = smpl_joints_2d,
                    pred_camera=pred_camera,
                    pred_joints_smpl_zerorot_zerobeta = smpl_output_zerorot_zerobeta.joints, # 1x49x3
                    pred_body_pose = pred_aa, # 1 x 72
                )]
                
                # prepare filter_settings
                filter_settings = EasyDict(
                    verbose=args.verbose,
                    use_openpose_joints=args.use_openpose_joints,
                    ignore_joints_outside=args.ignore_joints_outside,
                    freeze_foc=args.freeze_foc,
                    fovy=args.fovy,
                    lr=args.lr,
                    epochs=args.epochs,
                    thres=args.joint_loss_thres,
                    save_with_human_params=False,
                )

                # convert 'weak-perspective' to 'perspective'
                persp_cam_smpl_list, _ = apply_wp_to_p(
                    filter_settings=filter_settings,
                    mocap_output_list=mocap_output_list, 
                    img_size=img.shape[:2]
                )

                if persp_cam_smpl_list[0]['min_loss'] > args.reprojection_error_thresh:
                    if args.verbose:
                        print(f"continue eft-idx: {eft_idx} -> since high perspective reprojection error")
                    continue

                # add to data
                data['persp_cam_list'] = persp_cam_smpl_list
                data['pred_aa'] = pred_aa
                data['pred_betas'] = pred_betas
                data['smpl_joints_2d'] = smpl_joints_2d
            
            # visualize
            if args.visualize:
                plt.figure(figsize=(20,20))
                plt.suptitle(f"category:{category}/currently_filterd_and_saved:{num_filtered+1}/eft_idx:{eft_idx:09}")
                
                # show person mask
                plt.subplot(3,3,1); plt.imshow(data['mask']['person'])
                [x,y,x_,y_] = data['bbox']['person']
                # show person bbox & joints
                img_person = cv2.rectangle(img.copy(), (int(x), int(y)), (int(x_), int(y_)), (255,0,0), 5)
                plt.subplot(3,3,2); plt.imshow(img_person)
                # show joints
                for joint_idx in range(len(smpl_joints_2d)):
                    x, y = smpl_joints_2d[joint_idx]
                    plt.plot(x, y, marker="o", markersize=5, markeredgecolor="blue", markerfacecolor = "blue")

                # show object mask
                plt.subplot(3,3,3); plt.imshow(data['mask'][category])
                [x,y,x_,y_] = data['bbox'][category]
                # show object bbox & joints
                img_object = cv2.rectangle(img.copy(), (int(x), int(y)), (int(x_), int(y_)), (255,0,0), 5)
                plt.subplot(3,3,4); plt.imshow(img_object)
                
                human_idx = 0
                assert len(mocap_output_list) == len(persp_cam_smpl_list)
                # prepare results
                persp_cam_smpl = persp_cam_smpl_list[human_idx] # perspective camera
                R = torch.tensor(persp_cam_smpl['R']).to("cuda") # extrinsic: 4x3x3 or 3x3
                t = torch.tensor(persp_cam_smpl['t']).to("cuda") # extrinsic: 4x3 or 3,  
                K = torch.tensor(persp_cam_smpl['K']).to("cuda") # intrinsic & focal_length: 4x3x3 or 3x3
                if R.ndim != 3:
                    assert R.shape == (3,3)
                    assert t.shape == (3,)
                    assert K.shape == (3,3)
                    R = torch.stack([R]*4, dim=0)
                    t = torch.stack([t]*4, dim=0)
                    K = torch.stack([K]*4, dim=0)

                    global_zerorot_joint = persp_cam_smpl_list[human_idx]['joints_used_SMPL']
                    global_zerorot_joint_batch = np.stack([global_zerorot_joint]*4, axis=0)
                else:        
                    global_zerorot_joint_batch = persp_cam_smpl_list[human_idx]['joints_used_SMPL_batch'] # Joints in SMPL space
                focal_length = persp_cam_smpl['focal_length'] # intrinsic & focal_length: 4, or scalar
                # min_losses = persp_cam_smpl['min_losses'] # joint reprojection loss: 4, or scalar

                homo=torch.matmul(K,(torch.matmul(R, torch.tensor(global_zerorot_joint_batch).to("cuda").permute(0,2,1))\
                                    + t[:,:,None])).permute(0,2,1) # 4xJx3
                projected_joint_img_smpl = homo[:,:,:2].clone()
                projected_joint_img_smpl[:,:,0]/=homo[:,:,2] # Homogenize
                projected_joint_img_smpl[:,:,1]/=homo[:,:,2] # Homogenize
                projected_joint_img_smpl = projected_joint_img_smpl.detach().cpu().numpy() # 4xJx2
                # to write on subplot subtitles
                if 'idx2name' not in persp_cam_smpl_list[human_idx]:
                    smpl_idx2name = ['ONLY LearnedK-Zerobeta Done']
                else:
                    smpl_idx2name = persp_cam_smpl_list[human_idx]['idx2name']

                for i in range(len(smpl_idx2name)):
                    plt.subplot(3,3,i+5); plt.imshow(img)
                    plt.gca().set_title(smpl_idx2name[i])
                    # GT joints
                    for idx in range(len(smpl_joints_2d)):
                        x = smpl_joints_2d[idx][0]; y=smpl_joints_2d[idx][1]
                        plt.plot(x, y, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
                    # Projected joints
                    for idx in range(len(projected_joint_img_smpl[i])):
                        x = projected_joint_img_smpl[i][idx][0]; y = projected_joint_img_smpl[i][idx][1]
                        plt.plot(x, y, marker="o", markersize=5, markeredgecolor="orange", markerfacecolor="orange")
                plt.show()
                plt.close()

            ## update to 'eft_data_save'
            # assert
            if is_overridden: assert eft_idx in eft_data_save.keys()
            else: assert eft_idx not in eft_data_save.keys()
            
            # save & update 'num_filtered'
            eft_data_save[eft_idx] = data
            num_filtered += 1
            
            # update description
            desc = f"Extending COCO-EFT Dataset for [Category: '{category}'] --> Num-Filtered: ({num_filtered})"
            pbar.set_description(desc)

        if args.verbose:
            print(f"Num-filtered for {category} when min bbox iou is {args.min_bbox_overlap} --> {num_filtered}")       
        
        ## add statistics for given object category
        stats_per_category[category] = num_filtered

    # placeholder for 'results-to-save' extended-coco-eft dataset
    to_save = {
        'eft_data': list(eft_data_save.values()), 
        'stats': stats_per_category,
        'min_bbox_overlap': args.min_bbox_overlap,
        'args': args,
    }

    # print statistics
    if args.verbose: print(to_save['stats'])

    # save results
    with open(args.save_pth, "wb") as handle: pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+")

    parser.add_argument("--original_dataset_pth", type=str, default=COCO_EFT_VAL_PTH)
    parser.add_argument("--save_pth", default=EXTENDED_COCO_EFT_VAL_SAVE_PTH)

    parser.add_argument("--min_bbox_overlap", type=float, default=DEFAULT_EFT_MIN_BBOX_OVERLAP)
    parser.add_argument("--reprojection_error_thresh", type=float, default=DEFAULT_EFT_REPROJECTION_ERROR_THRESH)
    parser.add_argument("--use_openpose_joints", action="store_true")
    parser.add_argument("--ignore_joints_outside", action="store_true")
    parser.add_argument("--freeze_foc", action="store_true")
    parser.add_argument("--fovy", type=float, nargs="?", default=DEFAULT_EFT_FOV)
    parser.add_argument("--lr", type=float, default=DEFAULT_EFT_LR)
    parser.add_argument("--joint_loss_thres", type=float, default=DEFAULT_EFT_JOINT_LOSS_THRES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EFT_EPOCHS)

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--verbose", action='store_true')

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # seed for reproducible generation
    seed_everything(args.seed)

    # extend coco-eft dataset (validation-set) to use as test-set
    extend_coco_eft_dataset(args=args)