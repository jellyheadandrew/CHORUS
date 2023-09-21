import pickle5 as pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

import cv2

from utils.transformations import batch_rodrigues, get_focal_length, convert_Kparams_Kmat


def apply_wp_to_p(
        filter_settings, 
        img_pth=None, 
        human_pth=None, 
        mocap_output_list=None, 
        img_size=None,
        device='cuda'
    ):
    # image size
    if img_size is None: H, W = cv2.imread(img_pth).shape[:2]
    else: H, W = img_size

    # load mocap-output-list
    if mocap_output_list is None:
        with open(human_pth, "rb") as handle: 
            human_parameters = pickle.load(handle)
            mocap_output_list = [] if human_parameters == "NO HUMANS" else human_parameters["mocap_output_list"]

    # number of humans
    num_humans = len(mocap_output_list)
    if filter_settings['verbose']:
        print(f"\tNumber of predicted humans in this case: {num_humans}")
    
    # a list to save element (min_loss, R, t, K) 
    persp_cam_list = []

    # iterate for all humans
    for human_idx in range(num_humans):
        ## load input joints (3d-smpl) & target joints (2d-projection) for optimization
        # smpl joints rendered on image
        pred_joints_img=mocap_output_list[human_idx]['pred_joints_img'][:,:2] # -> Shape: (49,2), xy format

        # smpl joints are set as globally zero-rotation (aligned at center, same front direction)
        pred_joints_smpl=mocap_output_list[human_idx]['pred_joints_smpl_zerorot_zerobeta'] # (1,49,3), direct output of smpl model
        pred_joints_smpl = pred_joints_smpl[0]

        # if using openpose joints, use all 24+25 joints for target. if not, use 24 smpl joints for target.
        if not filter_settings['use_openpose_joints']:
            pred_joints_img = pred_joints_img[25:]
            pred_joints_smpl = pred_joints_smpl[25:]
            
        # if ignoring joints outside the image range [0,H] X [0,W], filter the target joints.
        if filter_settings['ignore_joints_outside']:
            joints_on_image_below = np.logical_and(pred_joints_img[:,0] < W, pred_joints_img[:,1] < H)
            joints_on_image_above = np.logical_and(pred_joints_img[:,0] > 0, pred_joints_img[:,1] > 0)
            joints_on_image = np.logical_and(joints_on_image_above, joints_on_image_below)
            pred_joints_img = pred_joints_img[joints_on_image]
            pred_joints_smpl = pred_joints_smpl[joints_on_image]
            
        # number of joints used for optimization
        num_target_joints = pred_joints_smpl.shape[0]

        # convert joints vector to tensor
        pred_joints_smpl_t = torch.tensor(pred_joints_smpl, device=device)
        pred_joints_img_t = torch.tensor(pred_joints_img, device=device)

        ## load focal length
        focal_length_init_t=torch.tensor(get_focal_length(1.0, filter_settings['fovy']))
        focal_length = nn.Parameter(focal_length_init_t, requires_grad=True)
        
        ## load translation
        # orthographic camera (scale, tx, ty) in smpl scale
        cams=mocap_output_list[human_idx]['pred_camera']

        # set initial translation
        t_init = torch.tensor([cams[1], cams[2], 2.0 * focal_length / cams[0]])
        t_param = nn.Parameter(t_init, requires_grad=True)
        
        # initialize rotation
        R_init = mocap_output_list[human_idx]['pred_body_pose'][0,:3] # (3,), global rotation in axis-angle (unnormalized)
        R_param = nn.Parameter(torch.Tensor(R_init), requires_grad=True)

        ## declare optimizer
        if filter_settings['freeze_foc']:
            optimizer = torch.optim.Adam([R_param, t_param], lr=filter_settings['lr'])
        else:
            optimizer = torch.optim.Adam([R_param, t_param, focal_length], lr=filter_settings['lr'])  

        ## optimize perspective camera
        min_loss = 1e10
        if filter_settings['verbose']: pbar = tqdm(range(filter_settings['epochs']))
        else: pbar = range(filter_settings['epochs'])
        
        for epoch in pbar:
            # clear optimizer grad
            optimizer.zero_grad()
            
            # prepare R, t, K
            R_w2c_rotmat = batch_rodrigues(R_param.unsqueeze(0)).squeeze().float().to(device)
            t_w2c = t_param.float().to(device)
            K_param = focal_length.float().to(device) * torch.Tensor([1,1,0,0]).to(device) + torch.Tensor([0,0,0.5,0.5 * H / W]).to(device)
            K_param = K_param * torch.Tensor([W]).to(device) # pixel-scale K 
            K = convert_Kparams_Kmat(K_param)

            # project smpl joints to image
            tmp=torch.matmul(K,(torch.matmul(R_w2c_rotmat, pred_joints_smpl_t.T)+t_w2c.expand(pred_joints_smpl_t.shape).T)).T
            projected_joint_img = tmp[:,:2].clone()
            projected_joint_img[:,0]/=tmp[:,2]
            projected_joint_img[:,1]/=tmp[:,2]

            # calculate l2-error between projected joints and target joints
            loss = torch.linalg.norm(pred_joints_img_t - projected_joint_img)
            loss = loss.divide(num_target_joints)

            # backward pass
            loss.backward()

            # whether early termination occurs
            if loss.item() < filter_settings['thres']:
                R_best = R_param.clone().detach()
                t_best = t_param.clone().detach()
                K_best = K_param.clone().detach()
                min_loss = loss.item()
                if filter_settings['verbose']:
                    print(f'Early termination with min_loss: {min_loss} and epoch: {epoch}')
                break

            # update the parameters
            optimizer.step()

            # update minimum loss
            if loss.item() < min_loss:
                    R_best = R_param.clone().detach()
                    t_best = t_param.clone().detach()
                    K_best = K_param.clone().detach()
                    min_loss = loss.item()

        # print when max epoch reached
        if filter_settings['verbose']:
            print(f'Reached max epoch: {epoch} with min_loss: {min_loss}')

        # (R, t, K) as convenient format (loss, SO(3) matrix, 3-dim vec, 3x3 matrix)
        R_best2save = batch_rodrigues(R_best.unsqueeze(0)).squeeze().float().to(device)
        t_best2save = t_best.float().to(device)
        K_best2save = convert_Kparams_Kmat(K_best)

        # save the results to list
        persp_cam_list.append(dict(
            min_loss=min_loss,
            R=R_best2save.detach().cpu().numpy(),
            t=t_best2save.detach().cpu().numpy(),
            K=K_best2save.detach().cpu().numpy(),
            joints_used_GT_image=pred_joints_img_t.detach().cpu().numpy(),
            joints_used_optimized_image=projected_joint_img.detach().cpu().numpy(),
            joints_used_SMPL=pred_joints_smpl_t.detach().cpu().numpy(),
            focal_length=focal_length.detach().cpu().numpy(),
            img_size=(H,W),
            filter_settings=filter_settings,                    
        ))
    
    # raise error if returned list has number-of-human mismatch
    assert len(persp_cam_list) == num_humans, f"There must be {num_humans} number of outputs, same as number of human in image."

    return persp_cam_list, (human_parameters if filter_settings['save_with_human_params'] else None)