import pickle5 as pickle

import torch

def load_aggr_input(
        cam_world_key,
        category_id,
        fps,
        zerobeta=True,
    ):
    # keys for fp
    cam_world_key = cam_world_key.lower()
    cam_key = f'cam_{cam_world_key.lower()}_pth'
    human_key = f'human_{cam_world_key.lower()}_pth'
    seg_key = 'human_smpl_pth' # post-processed 'seg' info is saved under here!

    # load cam
    with open(fps[cam_key], "rb") as handle:
        cam_data = pickle.load(handle)
        persp_cam_list = cam_data['persp_cam_list']

    # load smpl parameters
    with open(fps[human_key], "rb") as handle:
        human_parameters = pickle.load(handle)
        mocap_output_list = human_parameters['mocap_output_list']

    # load post-processed segmentations
    with open(fps[seg_key], "rb") as handle:
        human_parameters = pickle.load(handle)
        instances = human_parameters['auxiliary']['instances_including_object']
    
    # prepare cam_R, cam_t, cam_K, thetas, betas
    aggr_input_list = []
    for i in range(len(persp_cam_list)):
        # prepare segmentation
        is_class = instances.pred_classes == category_id
        mask = torch.any(instances.pred_masks[is_class], dim=0)

        # prepare cam
        cam_R = torch.tensor(persp_cam_list[i]['R'])
        cam_t = torch.tensor(persp_cam_list[i]['t'])
        cam_K = torch.tensor(persp_cam_list[i]['K'])

        # pose (thetas)
        pose = torch.tensor(mocap_output_list[i]['pred_body_pose'], dtype=torch.float32)
        betas = torch.zeros([10], dtype=torch.float32) if zerobeta else torch.tensor(mocap_output_list[i]['betas'], dtype=torch.float32)
        
        # add to 'input' list
        aggr_input_list.append(
            {
                'cam_R': cam_R,
                'cam_t': cam_t,
                'cam_K': cam_K,
                'mask': mask,
                'pose': pose,
                'betas': betas,
            }
        )

    return aggr_input_list