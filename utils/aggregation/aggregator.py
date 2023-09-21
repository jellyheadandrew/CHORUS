import os
from glob import glob
import shutil

import pickle5 as pickle
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict

import math
import numpy as np
import torch

import cv2
import matplotlib.pyplot as plt

import open3d as o3d
import mayavi.mlab as mlab
from tvtk.util import ctf

from smplx.lbs import blend_shapes, vertices2joints

from constants.mayavi_cams import VIDEO_CAMS

from utils.transformations import get_azimuth, batch_rodrigues, batch_rigid_transform, perspective_projection_grid
from utils.misc import to_np_torch_recursive
from utils.aggregation.load_lbs_weight import precompute_lbs_weights

from visualization.visualize_video import generate_video_from_imgs
from visualization.visualize_open3d import return_cube, return_coordinate_frame, return_triangle_mesh, return_camera_frustum



class Aggregator:
    def __init__(self, core: EasyDict):
        super().__init__()

        # core
        self.core = deepcopy(core)

        # whether to use torch or np & device
        self.use_torch = core.GLOBAL.use_torch
        self.device = core.GLOBAL.device

        ## cache for registering inputs for aggregation
        self.cache = EasyDict({_cam_world_key: EasyDict() for _cam_world_key in self.core.keys()}) 
        self.num_processed = EasyDict({_cam_world_key: 0 for _cam_world_key in self.core.keys()})


        ## full_aggregations: aggregating 'all' images (clustering disabled)
        # declare for each worlds
        self.full_aggregations = EasyDict({_cam_world_key: EasyDict() for _cam_world_key in self.core.keys()})
        del self.full_aggregations['GLOBAL']
        
        # add required variables for each world
        for _cam_world_key in self.full_aggregations.keys():
            # voting grid - mask occupancy
            self.full_aggregations[_cam_world_key].visual_hull = np.zeros(
                [
                    self.core[_cam_world_key].N_x, 
                    self.core[_cam_world_key].N_y, 
                    self.core[_cam_world_key].N_z,
                ], dtype=float
            )
            # voting grid - image occupancy
            self.full_aggregations[_cam_world_key].full_image_visual_hull = np.zeros(
                [
                    self.core[_cam_world_key].N_x,
                    self.core[_cam_world_key].N_y,
                    self.core[_cam_world_key].N_z,
                ], dtype=float
            )
            # camera-pose (R,t,K) history
            self.full_aggregations[_cam_world_key].aggr_input_history = {}
            # geometries for visualization (mayavi)
            self.full_aggregations[_cam_world_key].geometries = EasyDict(
                    grid=dict( # create grid using lines
                        length_x=self.core[_cam_world_key].length_x,
                        length_y=self.core[_cam_world_key].length_y,
                        length_z=self.core[_cam_world_key].length_z,
                        center=self.core[_cam_world_key].center,
                        color=self.core[_cam_world_key].grid_color,
                    ),
                    coordinate_frame=dict(size=1.0), # create coordinate frame
                    template = dict( # 3. create template mesh in canonical space
                        verts=self.core[_cam_world_key].v_template_holistic,
                        faces=self.core[_cam_world_key].v_template_faces_holistic,
                        color=self.core[_cam_world_key].v_template_color
                    ),
                )
            # geometries for visualization (open3d)
            self.full_aggregations[_cam_world_key].o3d_geometries = EasyDict(
                    grid=return_cube( # create grid using lines
                        length_x=self.core[_cam_world_key].length_x,
                        length_y=self.core[_cam_world_key].length_y,
                        length_z=self.core[_cam_world_key].length_z,
                        center=self.core[_cam_world_key].center,
                        color=self.core[_cam_world_key].grid_color,
                    ),
                    coordinate_frame=return_coordinate_frame(size=1.0), # create coordinate frame
                    template = return_triangle_mesh( # create template mesh in canonical space
                        verts=self.core[_cam_world_key].v_template_holistic,
                        faces=self.core[_cam_world_key].v_template_faces_holistic,
                        color=self.core[_cam_world_key].v_template_color
                    ),
                )
        

        ## aggregations: aggregating images (clustering enabled) in the specified world
        # declare for each worlds
        self.aggregations = EasyDict({_cam_world_key: EasyDict() for _cam_world_key in self.core.keys()})
        del self.aggregations['GLOBAL']

        # iterate for worlds
        for _cam_world_key in self.aggregations.keys():
            # dictionary of body_part_name: List[tag1, tag2, ...] OR None
            aggr_world_name_per_body_parts = self.core[_cam_world_key].aggr_world_name_per_body_parts 

            # each elements of dictionary would be world name
            self.aggregations[_cam_world_key].aggr_world_name_per_body_parts = {
                body_part_name: 
                    list(
                        set(
                            aggr_world_name_per_body_parts[body_part_name] + [_cam_world_key]
                        )
                    ) if aggr_world_name_per_body_parts[body_part_name] is not None 
                    else [_cam_world_key]
                for body_part_name in self.core[_cam_world_key].body_part_names
            }

            # each elements of dictionary would be voting grid (mask occupancy)
            self.aggregations[_cam_world_key].visual_hull_per_body_parts = {
                body_part_name: {
                    _aggr_world_key: np.zeros( # shape: NxNxN, dtype: float
                        [
                            self.core[_aggr_world_key].N_x, 
                            self.core[_aggr_world_key].N_y, 
                            self.core[_aggr_world_key].N_z,
                        ], dtype=float
                    )
                    for _aggr_world_key in self.aggregations[_cam_world_key].aggr_world_name_per_body_parts[body_part_name]
                } for body_part_name in self.core[_cam_world_key].body_part_names
            }

            # each elements of dictionary would be voting grid (image occupancy)
            self.aggregations[_cam_world_key].full_image_visual_hull_per_body_parts = {
                body_part_name: { 
                    _aggr_world_key: np.zeros( # shape: NxNxN, dtype: float
                        [ 
                            self.core[_aggr_world_key].N_x, 
                            self.core[_aggr_world_key].N_y, 
                            self.core[_aggr_world_key].N_z
                        ], dtype=float
                    )
                    for _aggr_world_key in self.aggregations[_cam_world_key].aggr_world_name_per_body_parts[body_part_name]
                } for body_part_name in self.core[_cam_world_key].body_part_names
            }
            
            # camera-pose (R,t,K) history
            self.aggregations[_cam_world_key].aggr_input_history_per_body_parts = {
                body_part_name: {
                    _aggr_world_key: EasyDict()
                    for _aggr_world_key in self.aggregations[_cam_world_key].aggr_world_name_per_body_parts[body_part_name]
                } for body_part_name in self.core[_cam_world_key].body_part_names
            }
            
            # geometries for visualization (mayavi)
            self.aggregations[_cam_world_key].geometries_per_body_parts = {
                body_part_name: {
                    _aggr_world_key: EasyDict(
                        grid=dict( # create Grid using lines
                            length_x=self.core[_aggr_world_key].length_x,
                            length_y=self.core[_aggr_world_key].length_y,
                            length_z=self.core[_aggr_world_key].length_z,
                            center=self.core[_aggr_world_key].center,
                            color=self.core[_aggr_world_key].grid_color,
                            ),
                        coordinate_frame=dict(size=1.0), # create coordinate frame
                        template = dict( # create Template in Canonical Frame
                            verts=self.core[_aggr_world_key].v_template_holistic,
                            faces=self.core[_aggr_world_key].v_template_faces_holistic,
                            color=self.core[_aggr_world_key].v_template_color
                        ),
                    ) for _aggr_world_key in self.aggregations[_cam_world_key].aggr_world_name_per_body_parts[body_part_name]
                } for body_part_name in self.core[_cam_world_key].body_part_names
            }
            # geometries for visualization (open3d)
            self.aggregations[_cam_world_key].o3d_geometries_per_body_parts = {
                body_part_name: {
                    _aggr_world_key: EasyDict(
                        grid=return_cube( # create Grid using lines
                            length_x=self.core[_aggr_world_key].length_x,
                            length_y=self.core[_aggr_world_key].length_y,
                            length_z=self.core[_aggr_world_key].length_z,
                            center=self.core[_aggr_world_key].center,
                            color=self.core[_aggr_world_key].grid_color,
                            ),
                        coordinate_frame=return_coordinate_frame(size=1.0), # create coordinate frame
                        template = return_triangle_mesh( # create Template in Canonical Frame
                            verts=self.core[_aggr_world_key].v_template_holistic,
                            faces=self.core[_aggr_world_key].v_template_faces_holistic,
                            color=self.core[_aggr_world_key].v_template_color
                        ),
                    ) for _aggr_world_key in self.aggregations[_cam_world_key].aggr_world_name_per_body_parts[body_part_name]
                } for body_part_name in self.core[_cam_world_key].body_part_names
            }
    
        # send all tensors to torch.Tensor or np.ndarray
        self.core = to_np_torch_recursive(self.core, use_torch=self.use_torch, device=self.device)
        self.full_aggregations = to_np_torch_recursive(self.full_aggregations, use_torch=self.use_torch, device=self.device)
        self.aggregations = to_np_torch_recursive(self.aggregations, use_torch=self.use_torch, device=self.device)
        


    ## using default settings from 'core'. hierarchy: GLOBAL > world name
    def query_from_core(self, name=None, _cam_world_key=None):
        if name in self.core.GLOBAL.keys():
            return self.core.GLOBAL[name]
        elif name in self.core[_cam_world_key].keys():
            return self.core[_cam_world_key][name]
        else:
            assert False, f"'{name}' does not exist in 'core.GLOBAL' & 'core.{_cam_world_key}'"


    ## left-to-right conversion functions
    def convert_left_smpl_to_right(self,cam_R, cam_t, cam_K, mask, pose, betas):
        """
            1. negate the y-rotation & z-rotation of 'left-(body part) angle'
            2. assign this negated 'left-(body part) angle' to 'right-(body part) angle'
        """
        # convert R (r_1, r_2, r_3 are column-vectors of R^-1)
        cam_R = cam_R.clone()
        cam_R[0,1] *= -1 # r_1's y
        cam_R[0,2] *= -1 # r_2's z
        cam_R[1,0] *= -1 # r_2's x
        cam_R[2,0] *= -1 # r_3's x
        # convert x-element of 't' (negation)
        cam_t = cam_t.clone() 
        cam_t[0] *= -1
        # convert width-dimension of 'mask'
        mask = mask.cpu().numpy()
        mask = mask[:,::-1].copy()
        mask = torch.tensor(mask, device=self.device)
        ## convert y-rotation & z-rotation of pose parameters
        assert pose.shape == (1,72), "smpl pose must be 48 in dimensions"
        pose = pose.clone()
        # negate all y-rotation & z-rotation
        pose[:,1::3] *= -1
        pose[:,2::3] *= -1
        # map 'left-(body part) angle' to 'right-(body part) angle' (reference: mmhuman3d.core.conventions.keypoints_mapping.KEYPOINTS_FACTORY)
        mapped_pose = torch.zeros_like(pose)
        mapped_pose[:,0:3] = pose[:,0:3]                # 1. pelvis <-> pelvis
        mapped_pose[:,3:6] = pose[:,6:9]                # 2. left-hip <-> right-hip
        mapped_pose[:,6:9] = pose[:,3:6]                # 3. left-hip <-> right-hip
        mapped_pose[:,9:12] = pose[:,9:12]              # 4. spine1 <-> spine1
        mapped_pose[:,12:15] = pose[:,15:18]            # 5. left-knee <-> right-knee
        mapped_pose[:,15:18] = pose[:,12:15]            # 6. left-knee <-> right-knee
        mapped_pose[:,18:21] = pose[:,18:21]            # 7. spine2 <-> spine2
        mapped_pose[:,21:24] = pose[:,24:27]            # 8. left-ankle <-> right-ankle
        mapped_pose[:,24:27] = pose[:,21:24]            # 9. left-ankle <-> right-ankle
        mapped_pose[:,27:30] = pose[:,27:30]            # 10. spine3 <-> spine3
        mapped_pose[:,30:33] = pose[:,33:36]            # 11. left-foot <-> right-foot
        mapped_pose[:,33:36] = pose[:,30:33]            # 12. left-foot <-> right-foot
        mapped_pose[:,36:39] = pose[:,36:39]            # 13. neck <-> neck
        mapped_pose[:,39:42] = pose[:,42:45]            # 14. left-collar <-> right-collar
        mapped_pose[:,42:45] = pose[:,39:42]            # 15. left-collar <-> right-collar
        mapped_pose[:,45:48] = pose[:,45:48]            # 16. head <-> head
        mapped_pose[:,48:51] = pose[:,51:54]            # 17. left-shoulder <-> right-shoulder
        mapped_pose[:,51:54] = pose[:,48:51]            # 18. left-shoulder <-> right-shoulder
        mapped_pose[:,54:57] = pose[:,57:60]            # 19. left-elbow <-> right-elbow
        mapped_pose[:,57:60] = pose[:,54:57]            # 20. left-elbow <-> right-elbow
        mapped_pose[:,60:63] = pose[:,63:66]            # 21. left-wrist <-> right-wrist
        mapped_pose[:,63:66] = pose[:,60:63]            # 22. left-wrist <-> right-wrist
        mapped_pose[:,66:69] = pose[:,69:72]            # 23. left-hand <-> right-hand
        mapped_pose[:,69:72] = pose[:,66:69]            # 24. left-hand <-> right-hand
        
        return cam_R, cam_t, cam_K, mask, mapped_pose, betas

    ## left-to-right conversion functions
    def convert_left_to_right(self, cam_world_key, cam_R, cam_t, cam_K, mask, pose, betas):
        converter = getattr(self, f"convert_left_{cam_world_key.lower()}_to_right")
        cam_R,cam_t,cam_K,mask,pose,betas = converter(cam_R,cam_t,cam_K,mask,pose,betas)

        # return as keyword-arguments
        return {
            'cam_R': cam_R,
            'cam_t': cam_t,
            'cam_K': cam_K,
            'mask': mask,
            'pose': pose,
            'betas': betas
        }
    

    ## preprocessing input
    def preprocess_input(self, cam_world_key, cam_R, cam_t, cam_K, mask, pose, betas):
        pose_prep_func = getattr(self, f"prepare_{cam_world_key.lower()}_pose_for_input")
        full_pose = pose_prep_func(pose)
                
        # betas as zeros
        betas = torch.zeros_like(betas)
        
        return {
            'cam_R': cam_R,
            'cam_t': cam_t,
            'cam_K': cam_K,
            'mask': mask,
            'pose': full_pose,
            'betas': betas
        }


    ## preparing smpl pose for input
    def prepare_smpl_pose_for_input(self, pose):
        # remove first 3 dimensions (global rotation)
        assert pose.shape[-1] in [72, 69] and (pose.shape[0] == 1 or pose.ndim == 1)
        pose = pose.clone().squeeze()
        if pose.shape == (72,):
            pose = pose[3:]
        if pose.shape == (69,):
            pose = pose[:]
        
        # batchify (1 x 69)
        pose = pose.unsqueeze(0)

        # global orient as 0
        global_orient = torch.zeros([1,3]).to(pose.dtype).to(pose.device)
        
        full_pose = torch.cat([global_orient, pose], dim=-1) # 1x72
        
        return full_pose
        

    ## register single image (cam, body pose) for 'later-aggregation'
    def register_cam_pose(
        self,
        cam_world_key, # world that conducts aggregation
        pose,
        betas,
        cam_R,
        cam_t,
        cam_K,
        mask,
        add_human_for_vis,
        auxiliary_geometry=EasyDict(),
        frustum_size=None,
        use_normalized_K=None,
        cam_color=None,
        tag_auxiliary:bool=None,
        auxiliary_exist_ok:bool=None, # raise error if auxiliary member of same key already exists in cache
        max_cam_save_num:int=None,
    ):
        ## 'key' for registering to cache
        cache_key = f'cache:{len(self.cache[cam_world_key].keys()):05}'

        ## add input-information to cache
        self.cache[cam_world_key][cache_key] = {
            'cam_world_key': cam_world_key, 
            'pose': pose, 
            'betas': betas,
            'cam_R': cam_R, 
            'cam_t': cam_t, 
            'cam_K': cam_K, 
            'mask': mask,
            'add_human_for_vis': add_human_for_vis,
            'auxiliary_geometry': auxiliary_geometry,
            'frustum_size': self.query_from_core('frustum_size', cam_world_key) if frustum_size is None else frustum_size,
            'use_normalized_K': self.query_from_core('use_normalized_K', cam_world_key) if use_normalized_K is None else use_normalized_K,
            'cam_color': self.query_from_core('cam_color', cam_world_key) if cam_color is None else cam_color, 
            'tag_auxiliary': self.query_from_core('tag_auxiliary', cam_world_key) if tag_auxiliary is None else tag_auxiliary,
            'auxiliary_exist_ok': self.query_from_core('auxiliary_exist_ok', cam_world_key) if auxiliary_exist_ok is None else auxiliary_exist_ok,
            'max_cam_save_num': self.query_from_core('max_cam_save_num', cam_world_key) if max_cam_save_num is None else max_cam_save_num
        }


    ## camera-weighting methods
    def return_camera_weights(self, method):
        ## dict to save 'camera-weights'
        camera_weights_per_world = dict()

        ## method 1: total-uniform
        if method == "total-uniform":
            for _cam_world_key in self.cache.keys():
                # declare dictionary to save camera-weight for cameras in world
                camera_weights_per_world[_cam_world_key] = dict()
                # iterate for 'registered inputs' in cache
                for cache_key in self.cache[_cam_world_key].keys():
                    camera_weights_per_world[_cam_world_key][cache_key] = 1.

        ## method 2: divide by 3 azimuth-regions and uniform 'within region'
        elif method == "azimuth-uniform":
            for _cam_world_key in self.cache.keys():
                # declare dictionary to save camera-weight for cameras in world
                camera_weights_per_world[_cam_world_key] = dict()
                # 60 degrees delimited azimuth space
                azim_range_cams = {0: [], 1: [], 2: []}
                # iterate for 'registered inputs' in cache
                for cache_key in self.cache[_cam_world_key].keys():
                    # find camera's location of origin
                    cam_R = self.cache[_cam_world_key][cache_key]['cam_R']
                    cam_t = self.aggregations[_cam_world_key][cache_key]['cam_t']
                    
                    # x, z position of camera
                    location = -cam_R.T @ cam_t
                    x = location[0]; z = location[2]

                    # check azimuth & register cam to 'azimuth-region'
                    sqrt_3 = math.sqrt(3)
                    if z>= sqrt_3 * x and z>=-sqrt_3 * x:
                        azim_range_cams[0].append(cache_key)
                    elif z<sqrt_3 * x and z<-sqrt_3 * x:
                        azim_range_cams[0].append(cache_key)
                    elif z >= 0 and z < sqrt_3 * x:
                        azim_range_cams[1].append(cache_key)
                    elif z < 0 and z >= sqrt_3 * x:
                        azim_range_cams[1].append(cache_key)
                    elif z >= 0 and z<-sqrt_3 * x:
                        azim_range_cams[2].append(cache_key)
                    elif z < 0 and z >= -sqrt_3 * x:
                        azim_range_cams[2].append(cache_key)
                    else:
                        assert False

                # calculate camera-weights for '_cam_world_key'
                weight = dict()
                weight[0] = 1. / len(azim_range_cams[0]) if len(azim_range_cams[0]) > 0 else 1.
                weight[1] = 1. / len(azim_range_cams[1]) if len(azim_range_cams[1]) > 0 else 1.
                weight[2] = 1. / len(azim_range_cams[2]) if len(azim_range_cams[2]) > 0 else 1.

                # update 'camera-weight' per world'
                for azim_region_key in azim_range_cams.keys():
                    for cache_key in azim_range_cams[azim_region_key]:
                        assert cache_key not in camera_weights_per_world[_cam_world_key].keys(), "Duplicate keys."
                        camera_weights_per_world[_cam_world_key][cache_key] = weight[azim_region_key]
                        
        elif method == "azimuth-uniform-fine":
            N = 12
            d_theta = 360. / N
            for _cam_world_key in self.cache.keys():
                # declare dictionary to save camera-weight for cameras in world
                camera_weights_per_world[_cam_world_key] = dict()
                # azimuth regions
                azim_range_cams = {i: [] for i in range(N)}
                # iterate for 'registered inputs' in cache
                for cache_key in self.cache[_cam_world_key].keys():
                    # find camera's location of origin
                    cam_R = self.cache[_cam_world_key][cache_key]['cam_R']
                    cam_t = self.cache[_cam_world_key][cache_key]['cam_t']
                    # x, z position of camera
                    location = -cam_R.T @ cam_t
                    x = location[0]; z = location[2]

                    # check azimuth & register cam to 'azimuth-region'
                    azimuth = get_azimuth(x,z)
                    i = int(azimuth / d_theta) % N
                    azim_range_cams[i].append(cache_key)

                # calculate camera-weights for '_cam_world_key'
                weight = dict()
                for i in range(N):
                    weight[i] = 1. / len(azim_range_cams[i]) if len(azim_range_cams[i]) > 0 else 1.

                # update 'camera-weight' per world'
                for azim_region_key in azim_range_cams.keys():
                    for cache_key in azim_range_cams[azim_region_key]:
                        assert cache_key not in camera_weights_per_world[_cam_world_key].keys(), "Duplicate keys."
                        camera_weights_per_world[_cam_world_key][cache_key] = weight[azim_region_key]

        else:
            assert False, "Invalid camera-sampling method."

        return camera_weights_per_world
    

    ## aggregate all images (cam/body pose) in 'cache'
    def aggregate_all(self, camera_sampling="all", verbose=False):
        # 'camera-weighting' method
        camera_weights_per_world = self.return_camera_weights(method=camera_sampling) # camera-weights per world
        # print message
        for _cam_world_key in self.cache.keys():
            if verbose:
                print(f"==== aggregating {len(self.cache[_cam_world_key].keys())} registered inputs in cache ====")
            
            # iterate through registered inputs in cache
            for cache_key, aggr_input in self.cache[_cam_world_key].items():
                # prepare 'aggr_input' as torch/numpy
                aggr_input = to_np_torch_recursive(aggr_input, use_torch=self.use_torch, device=self.device)
                # run visual hull
                self.aggregate_single_input(
                    camera_weight=camera_weights_per_world[_cam_world_key][cache_key],
                    verbose=verbose,    
                    **aggr_input
                )
                # print which registered-input is aggregated 
                if verbose:
                    print(f"\taggregation for {cache_key} done...")

            # re-declare 'clean' cache after aggregation
            self.cache[_cam_world_key] = EasyDict()

            # print success for aggregation
            if verbose:
                print(f"==== aggregation for '{_cam_world_key}' succesful ====")


    ## apply projection and remove if outside image
    def retain_projection_inside_img(self, projected_grid, img_size):
        # unpack 'img_size'
        H, W = img_size

        # round-down to integer-values (to obtain pixel-values) 
        projected_grid = projected_grid.type(torch.long) # Shape: [2,*grid_size]
        
        # remove indices if 'projected-points' are outside the image
        above_zero = torch.logical_and(projected_grid[0] >= 0, projected_grid[1] >= 0)
        below_img_size = torch.logical_and(projected_grid[0] < W, projected_grid[1] < H)
        inside_img = torch.logical_and(above_zero, below_img_size) # Shape: [*grid_size]
        projected_grid = projected_grid[:,inside_img] # Shape: [2, *grid_size]

        return projected_grid, inside_img


    ## save mask occupancy & image occupancy per 'voxel in grid'
    def uplift_mask_image_occupancy(self, projected_grid, mask, img_size):
        # remove indices if 'projected-points' are outside the image
        projected_grid, inside_img = self.retain_projection_inside_img(projected_grid, img_size=img_size)

        # check mask-occupancy of 'projected-points'
        mask_occupancy_grid = torch.zeros_like(inside_img, dtype=torch.float32)
        mask_occupancy_grid[inside_img] = mask[projected_grid[1], projected_grid[0]].type(torch.float32) # Shape: [N_x, N_y, N_z] / dtype: long
        # check image-occupancy of 'projected-points'
        image_occupancy_grid = torch.zeros_like(inside_img, dtype=torch.float32)
        image_occupancy_grid[inside_img] = torch.ones_like(mask)[projected_grid[1], projected_grid[0]].type(torch.float32) # Shape: [N_x, N_y, N_z] / dtype: long

        return mask_occupancy_grid, image_occupancy_grid, inside_img


    ## aggregate single input given
    def aggregate_single_input(
        self,
        camera_weight,
        verbose, 
        cam_world_key,
        cam_R,
        cam_t,
        cam_K,
        mask,
        pose,
        betas,
        add_human_for_vis,
        frustum_size,
        use_normalized_K,
        cam_color,
        auxiliary_geometry,
        tag_auxiliary,
        auxiliary_exist_ok,
        max_cam_save_num,
    ):
        # mask shape
        H, W = mask.shape
        
        # camera-intrinsics
        if use_normalized_K:
            focal_length = cam_K[0,0]
        else:
            focal_length = cam_K[0,0] / W

        # remove batch-dimension for 'pose/betas'
        pose = pose.squeeze(); betas = betas.squeeze()

        ## we don't need to re-compute aggregation result (mask-occupancy / image-occupancy) for every body-part
        mask_occupancy_precomputed = {
            (_cam_world_key, _aggr_world_key): None 
            for _aggr_world_key in self.aggregations.keys()
            for _cam_world_key in self.aggregations.keys()
        }
        image_occupancy_precomputed = {
            (_cam_world_key, _aggr_world_key): None 
            for _aggr_world_key in self.aggregations.keys()
            for _cam_world_key in self.aggregations.keys()
        }
        deformed_human_mesh_precomputed = {
            (_cam_world_key, _aggr_world_key): None
            for _aggr_world_key in self.aggregations.keys()
            for _cam_world_key in self.aggregations.keys()
        }

        ## run aggregation for all inputs [_cam_world_key -> body_part_name -> _aggr_world_key]
        for _cam_world_key in self.aggregations.keys():
            # update number of 'cameras done'
            cam_index = self.num_processed[_cam_world_key]
            self.num_processed[_cam_world_key] += 1
                        
            ## iterate for 'body parts'
            for body_part_name in self.core[_cam_world_key].body_part_names:
                # run visual hull for all [world-cluster-_aggr_world_key] grids
                for _aggr_world_key in self.aggregations[_cam_world_key].aggr_world_name_per_body_parts[body_part_name]:
                    # skip if not related for aggregation
                    if _aggr_world_key != cam_world_key and _cam_world_key != cam_world_key:
                        continue
                    
                    # use precomputed visual hull results (if exists)
                    if mask_occupancy_precomputed[(_cam_world_key, _aggr_world_key)] is not None:
                        # load visual hull and accum visual hull
                        mask_occupancy_grid = mask_occupancy_precomputed[(_cam_world_key, _aggr_world_key)]
                        image_occupancy_grid = image_occupancy_precomputed[(_cam_world_key, _aggr_world_key)]
                        deformed_human_mesh = deepcopy(deformed_human_mesh_precomputed[(_cam_world_key, _aggr_world_key)])

                    # precompute visual hull results (if not exists)
                    else:    
                        # print progress
                        if verbose:
                            print(f"precomputing 'visual-hull' & 'full-image-visual-hull' for '_cam_world_key': {_cam_world_key} / '_aggr_world_key': {_aggr_world_key} (given 'cam_world_key': {cam_world_key})")
                         
                        ## translations before/after 'skinning-deformation'
                        translations = dict(
                            smpl2cam_trans = (-1.) * self.core[_cam_world_key].canoncam2smpl_trans, # 1x3 or None
                            aggr2smpl_trans = self.core[_aggr_world_key].canoncam2smpl_trans, # 1x3 or None
                            deformed2world_trans = None,
                        )

                        ## canonical-grid for 'aggregation' (aggregation world)
                        aggr_canon_grid = self.core[_aggr_world_key].canon_grid.clone()
                        N_x, N_y, N_z = aggr_canon_grid.shape[1:]
                        
                        ## situation 1: '(canonical) aggregation world' -> '(pose-deformed) aggregation world' (same as camera-world)
                        if _aggr_world_key == cam_world_key:
                            # translate grid in 'aggregation world' before deformation (no translation)
                            pre_translation = torch.zeros([3], dtype=torch.float32).to(self.device)
                            canon_grid_pre_transform = aggr_canon_grid + pre_translation[:,None,None,None] # 3 x N x N x N
                            # forward deformation at 'aggregation world': [N_all_voxels x 3] X [(3*J)] => [N_all_voxels x 3] X [V x 3, F x 3]
                            deformed_canon_grid, deformed_human_mesh, post_translation = self.skinning_deformation(
                                cam_world_key,
                                canon_grid_pre_transform.reshape(3,-1).T, 
                                pose,
                                betas,
                                return_human=add_human_for_vis,
                            )
                            # translate grid in '(deformed) aggregation world'
                            cam_world_grid = deformed_canon_grid + post_translation
                            

                        # situation 2: 'aggregation world' -> '(canonical) camera world' -> '(pose-deformed) camera world'
                        elif _cam_world_key == cam_world_key:
                            # translate grid in 'aggregation world' to 'camera world' before deformation
                            pre_translation = translations['aggr2smpl_trans'] + translations['smpl2cam_trans']
                            canon_grid_pre_transform = aggr_canon_grid + pre_translation[:,None,None,None] # [3 x N x N x N]
                            # forward deformation at 'camera world': [N_all_voxels x 3] X [(3*J)] => [N_all_voxels x 3] X [V x 3, F x 3]
                            deformed_canon_grid, deformed_human_mesh, post_translation = self.skinning_deformation( 
                                cam_world_key,
                                canon_grid_pre_transform.reshape(3,-1).T,
                                pose, 
                                betas,
                                return_human=add_human_for_vis,
                            )
                            # translate grid in '(deformed) camera world'
                            cam_world_grid = deformed_canon_grid + post_translation


                        ## project 'deformed-grid' (in camera-world) into camera
                        projected_grid = perspective_projection_grid(
                            grid=cam_world_grid, 
                            K=cam_K, 
                            R=cam_R, 
                            t=cam_t, 
                            grid_size=(N_x, N_y, N_z), 
                            use_normalized_K=use_normalized_K, 
                            img_size=(H,W)
                        ) # Shape: 2 x N_x x N_y x N_z
                        

                        ## check 'image-occupancy' and 'mask-occupancy'
                        mask_occupancy_grid, image_occupancy_grid, _ = self.uplift_mask_image_occupancy(projected_grid, mask, img_size=(H,W))

                        ## save to 'precompute'
                        mask_occupancy_precomputed[(_cam_world_key, _aggr_world_key)] = mask_occupancy_grid.clone()
                        image_occupancy_precomputed[(_cam_world_key, _aggr_world_key)] = image_occupancy_grid.clone()
                        deformed_human_mesh_precomputed[(_cam_world_key, _aggr_world_key)] = deepcopy(deformed_human_mesh)

                    ## check if mask-rays pass 'interaction region' (N_inter_points x 3, LongTensor)
                    # voxels within 'interaction-region'
                    interaction_regions = self.core[_cam_world_key].interaction_regions[body_part_name][_aggr_world_key]
                    
                    # does mask-rays pass 'interaction region (pass)'?
                    dont_aggregate_for_body_part = False
                    for indices in interaction_regions['pass']:
                        interaction_region_mask_occupancy = mask_occupancy_grid[indices[:,0], indices[:,1], indices[:,2]].type(torch.long) # N_interaction_points
                        if torch.sum(interaction_region_mask_occupancy) == 0:
                            dont_aggregate_for_body_part = True
                            break
                    
                    # does mask-rays avoid 'interaction region (block)'?
                    for indices in interaction_regions['block']:
                        interaction_region_mask_occupancy = mask_occupancy_grid[indices[:,0], indices[:,1], indices[:,2]].type(torch.long) # N_interaction_points
                        if torch.sum(interaction_region_mask_occupancy) > 0:
                            dont_aggregate_for_body_part = True
                            break
                    
                    # if 'interaction-region' does not lie in 'mask-occupancy', continue
                    if dont_aggregate_for_body_part:
                        if verbose:
                            print(f"\tcontinue body_part: '{body_part_name}' for aggregation")
                        continue


                    ## aggregate mask-occupancy & image-occupancy
                    self.aggregations[_cam_world_key].visual_hull_per_body_parts[body_part_name][_aggr_world_key] += mask_occupancy_grid * camera_weight
                    self.aggregations[_cam_world_key].full_image_visual_hull_per_body_parts[body_part_name][_aggr_world_key] += image_occupancy_grid * camera_weight

                    ## add 'inputs' to history
                    if len(self.aggregations[_cam_world_key].aggr_input_history_per_body_parts[body_part_name][_aggr_world_key]) < max_cam_save_num:
                        # key for saving to history
                        history_key = f'cam:{cam_index:05}'
                        
                        # print which 'aggr_input' you register to history
                        if verbose:
                            print(f"\t{body_part_name}: {len(self.aggregations[_cam_world_key].aggr_input_history_per_body_parts[body_part_name][_aggr_world_key])}")
                        
                        # add 'aggr_input' to history
                        self.aggregations[_cam_world_key].aggr_input_history_per_body_parts[body_part_name][_aggr_world_key][history_key] = {
                            'pose': pose.clone().squeeze().detach().cpu().numpy(), # 3*J
                            'betas': betas.clone().squeeze().detach().cpu().numpy(), # 10
                            'cam_R': cam_R.clone().squeeze().detach().cpu().numpy(), # 3x3
                            'cam_t': cam_t.clone().squeeze().detach().cpu().numpy(), # 3,
                            'cam_K': cam_K.clone().squeeze().detach().cpu().numpy(), # 3x3
                            'mask': mask.clone().squeeze().detach().cpu().numpy(), # HxW
                        }

                        # add camera frustum
                        self.aggregations[_cam_world_key].o3d_geometries_per_body_parts[body_part_name][_aggr_world_key][history_key] = return_camera_frustum(
                            frustum_size=frustum_size, 
                            focal_length=focal_length.detach().cpu().numpy(), 
                            cam_R=cam_R.detach().cpu().numpy(), 
                            cam_t=cam_t.detach().cpu().numpy(), 
                            color=cam_color
                        )
                        self.aggregations[_cam_world_key].geometries_per_body_parts[body_part_name][_aggr_world_key][f'cam:{cam_index:05}'] = {
                            'frustum_size': frustum_size, 
                            'focal_length': focal_length.detach().cpu().numpy(), 
                            'cam_R': cam_R.detach().cpu().numpy(), 
                            'cam_t': cam_t.detach().cpu().numpy(), 
                            'color': cam_color
                        }
                            
                        # add human for visualization
                        if add_human_for_vis:
                            # add human-mesh
                            self.add_auxiliary_geometry(
                                _cam_world_key=_cam_world_key, 
                                body_part_name=body_part_name, 
                                _aggr_world_key=_aggr_world_key,
                                name = f"posed:{cam_index:05}",
                                geometry = deformed_human_mesh,
                                exist_ok=auxiliary_exist_ok,
                                tag_auxiliary=tag_auxiliary,
                            )

                    else:
                        self.aggregations[_cam_world_key].aggr_input_history_per_body_parts[body_part_name][_aggr_world_key][f'cam:{cam_index:05}'] = None
                        self.aggregations[_cam_world_key].o3d_geometries_per_body_parts[body_part_name][_aggr_world_key][f'cam:{cam_index:05}'] = None
                        self.aggregations[_cam_world_key].geometries_per_body_parts[body_part_name][_aggr_world_key][f'cam:{cam_index:05}'] = None

                        # add human-mesh
                        if add_human_for_vis:
                            self.add_auxiliary_geometry(
                                _cam_world_key=_cam_world_key, 
                                body_part_name=body_part_name, 
                                _aggr_world_key=_aggr_world_key,
                                name = f"posed:{cam_index:05}",
                                geometry = None, 
                                exist_ok=auxiliary_exist_ok,
                                tag_auxiliary=tag_auxiliary,
                            )                            
                    
                    # add manual auxiliary geometry
                    for name, geometry in auxiliary_geometry.items():
                        self.add_auxiliary_geometry(
                            _cam_world_key=_cam_world_key, 
                            body_part_name=body_part_name, 
                            _aggr_world_key=_aggr_world_key,
                            name = name,
                            geometry = deepcopy(geometry),
                            exist_ok=auxiliary_exist_ok,
                            tag_auxiliary=tag_auxiliary,
                        )

        ## for 'full_aggregations', aggregate no matter what body-part
        _cam_world_key = cam_world_key
        # Load Accum Visual Hull
        mask_occupancy_grid = mask_occupancy_precomputed[(_cam_world_key, _cam_world_key)]
        image_occupancy_grid = image_occupancy_precomputed[(_cam_world_key, _cam_world_key)]
        deformed_human_mesh = deepcopy(deformed_human_mesh_precomputed[(_cam_world_key, _cam_world_key)])
        assert mask_occupancy_grid is not None
        assert image_occupancy_grid is not None

        self.full_aggregations[cam_world_key].visual_hull += mask_occupancy_grid * camera_weight
        self.full_aggregations[cam_world_key].full_image_visual_hull += image_occupancy_grid * camera_weight
        assert f'cam_pose:{cam_index:05}' not in self.full_aggregations[cam_world_key].aggr_input_history.keys(), "Duplicate keys."
        if len(self.full_aggregations[cam_world_key].aggr_input_history) < max_cam_save_num:
            self.full_aggregations[cam_world_key].aggr_input_history[f'cam:{cam_index:05}'] = {
                'pose': pose.clone().squeeze().detach().cpu().numpy(), # 3*J
                'betas': betas.clone().squeeze().detach().cpu().numpy(), # 10
                'cam_R': cam_R.clone().squeeze().detach().cpu().numpy(), # 3x3
                'cam_t': cam_t.clone().squeeze().detach().cpu().numpy(), # 3,
                'cam_K': cam_K.clone().squeeze().detach().cpu().numpy(), # 3x3
                'mask': mask.clone().squeeze().detach().cpu().numpy(), # HxW
            }
            self.full_aggregations[cam_world_key].o3d_geometries[f'cam:{cam_index:05}'] = \
                return_camera_frustum(
                    frustum_size=frustum_size, 
                    focal_length=focal_length.detach().cpu().numpy(), 
                    cam_R=cam_R.detach().cpu().numpy(), 
                    cam_t=cam_t.detach().cpu().numpy(), 
                    color=cam_color
                )
            self.full_aggregations[cam_world_key].geometries[f'cam:{cam_index:05}'] = \
                dict(
                    frustum_size=frustum_size, 
                    focal_length=focal_length.detach().cpu().numpy(), 
                    cam_R=cam_R.detach().cpu().numpy(), 
                    cam_t=cam_t.detach().cpu().numpy(), 
                    color=cam_color
                )
            # add human if needed
            if add_human_for_vis:
                assert deformed_human_mesh is not None
                self.add_auxiliary_geometry(
                    _cam_world_key=cam_world_key,
                    body_part_name=None, 
                    _aggr_world_key=None,
                    name = f"posed:{cam_index:05}",
                    geometry = deformed_human_mesh, 
                    exist_ok=auxiliary_exist_ok,
                    tag_auxiliary=tag_auxiliary,
                    )

            # add manual auxiliary geometry
            for name, geometry in auxiliary_geometry.items():
                self.add_auxiliary_geometry(
                    _cam_world_key=cam_world_key, 
                    body_part_name=None, 
                    _aggr_world_key=None,
                    name = name,
                    geometry = deepcopy(geometry),
                    exist_ok=auxiliary_exist_ok,
                    tag_auxiliary=tag_auxiliary,
                    )                            
        else:
            self.full_aggregations[cam_world_key].aggr_input_history[f'cam:{cam_index:05}'] = None
            self.full_aggregations[cam_world_key].o3d_geometries[f'cam:{cam_index:05}'] = None
            self.full_aggregations[cam_world_key].geometries[f'cam:{cam_index:05}'] = None
            # add human if needed
            if add_human_for_vis:
                assert deformed_human_mesh is not None
                self.add_auxiliary_geometry(
                    _cam_world_key=_cam_world_key, 
                    body_part_name=None, 
                    _aggr_world_key=None,
                    name = f"posed:{cam_index:05}",
                    geometry = None, 
                    exist_ok=auxiliary_exist_ok,
                    tag_auxiliary=tag_auxiliary,
                )


    ## skinning-deformation using lbs-weights
    def skinning_deformation(
        self, 
        cam_world_key,
        points, # to deform, (Nx3)
        pose, # (Jx3),
        betas, # (10,)
        pose2rot: bool = True, 
        return_human: bool = False,
        backward: bool = False,
        backward_lbs_weight_grid=None,
        ):
        # forward-skinning / backward-skinning
        if backward_lbs_weight_grid is None:
            lbs_weight_grid = self.core[cam_world_key].lbs_weight_grid.clone()[None] # 1 x N_x x N_y x N_z
        else:
            lbs_weight_grid = backward_lbs_weight_grid

        # [N_all_voxels x 3] X [72] => [3 x N_all_voxels] and [Deformed Template] and [Dict of Translations]
        deformation_func = getattr(self, f"skinning_deformation_{cam_world_key.lower()}")
        deformed_grid, deformed_human_mesh, J_transformed = deformation_func(
            points=points, 
            pose=pose, 
            betas=betas, 
            pose2rot=pose2rot, 
            return_human=return_human,
            lbs_weight_grid=lbs_weight_grid,
            backward=backward, 
        )
        deformed2world_trans = torch.zeros([3], dtype=points.dtype, device=points.device)
        return deformed_grid, deformed_human_mesh, deformed2world_trans


    ## lbs-skinning for smpl
    def skinning_deformation_smpl(
            self, 
            points, 
            pose, 
            betas, 
            pose2rot, 
            return_human,
            lbs_weight_grid,
            backward:bool = False,
        ):
        """ [N_all_voxels x 3] X [72] -> [3 x N_all_voxels] and [Current Human] """
        # batchify inputs
        betas = betas.unsqueeze(dim=0).to(self.device) # 1 x 10
        pose = pose.unsqueeze(dim=0).to(self.device) # 1 x 72
        points = points.unsqueeze(dim=0).to(self.device) # 1 x N_points x 3
        N_x, N_y, N_z, N_joint = self.core['SMPL'].lbs_weight_grid.shape
                        
        # apply 'blend-skinning' and get joints of 'zero-pose' template mesh
        v_shaped = self.core.GLOBAL.smpl_info['v_template_T'] + blend_shapes(betas, self.core.GLOBAL.smpl_info['shapedirs'])
        J_tpose_movedbybeta = vertices2joints(self.core.GLOBAL.smpl_info['J_regressor'], v_shaped)
        
        ## compute 'pose-blend' offsets
        # identity
        identity = torch.eye(3).to(self.device) # Shape: 3
        
        # if we use angle-axis for pose
        if pose2rot:
            # joint-pose as rotation matrices
            rot_mats = batch_rodrigues(pose.view(-1, 3)).view([1, -1, 3, 3]) # Shape: 1 x (J+1) x 3 x 3
            # pose-blend features
            pose_feature = (rot_mats[:, 1:, :, :] - identity).view([1, -1]) # Shape: 1 x 9J
            # pose-blend offsets
            pose_offsets = torch.matmul(pose_feature, self.core.GLOBAL.smpl_info['posedirs']).view(1, -1, 3) # Shape: (1 x 9J) x (9J x 3V) -> 1 x 3V -> 1 x V x 3
        
        # if we use rotation-matrix for pose
        else:
            # joint-pose as rotation matrices
            rot_mats = pose.view(1, -1, 3, 3) # Shape: 1 x (J+1) x 3 x 3
            # pose-blend features
            pose_feature = (rot_mats[:, 1:, :, :] - identity).view([1, -1]) # Shape: 1 x 9J
            # pose-blend offsets
            pose_offsets = torch.matmul(pose_feature, self.core.GLOBAL.smpl_info['posedirs']).view(1, -1, 3) # Shape: (1 x 9J) x (9J x 3V) -> 1 x 3V -> 1 x V x 3
                        
        ## add 'pose-blend' offsets to 'shape-blend' + 'template'
        v_posed = pose_offsets + v_shaped.to(self.device)
        
        ## get transformation matrices (for all joints)
        J_transformed, A_tpose2current = batch_rigid_transform(
            rot_mats=rot_mats,
            joints=J_tpose_movedbybeta, 
            parents=self.core.GLOBAL.smpl_info['parents'], 
            dtype=torch.float32
        )
        
        ## calculate 'A' (transformation matrices) is for 'user-defined-template space -> t-pose-template space'
        # if pre-computed, use the 'A' value
        if 'A_custom2tpose' in self.core.GLOBAL.smpl_info.keys():
            # 'A' (transformation matrices) for 't-pose-template space' -> 'user-defined-template space'
            A_custom2tpose = self.core.GLOBAL.smpl_info['A_custom2tpose'] # 1 x (J+1) x 4 x 4
            # 'A' (transformation matrices) for 't-pose-template space' -> 'camera space' 
            A_custom2current = A_tpose2current @ A_custom2tpose # 1 x (J+1) x 4 x 4
        
        # if not pre-computed, run pre-computation if 'user-defined zero_pose' is not entirely 0
        elif self.core.GLOBAL.smpl_info['zero_pose'].type(torch.bool).any():
            # if we use angle-axis for pose
            if pose2rot:            
                # joint-pose as rotation matrices
                rot_mats_custom2tpose = batch_rodrigues(self.core.GLOBAL.smpl_info["zero_pose"].view(-1, 3)).view([1, -1, 3, 3]) # Shape: 1 x (J+1) x 3 x 3 
            else:
                rot_mats_custom2tpose = pose.view(1, -1, 3, 3) # 1 x (J+1) x 3 x 3
            
            # joint-locations for 't-pose template'
            J_tpose = vertices2joints(self.core.GLOBAL.smpl_info['J_regressor'], self.core.GLOBAL.smpl_info['v_template_T'])

            # 'A' (transformation matrices) for 't-pose-template space' -> 'user-defined-template space'
            _, A_tpose2custom = batch_rigid_transform(
                rot_mats=rot_mats_custom2tpose, 
                joints=J_tpose, 
                parents=self.core.GLOBAL.smpl_info['parents'], 
                dtype=torch.float32
            )

            # 'A' (transformation matrices) for 'user-defined-template space' -> 't-pose-template space'
            A_custom2tpose = torch.inverse(A_tpose2custom) # 1 x (J+1) x 4 x 4
            # update precomputed 'A' to core
            self.core.GLOBAL.smpl_info["A_custom2tpose"] = A_custom2tpose
            # 'A' (transformation matrices) for 't-pose-template space' -> 'camera space' 
            A_custom2current = A_tpose2current @ A_custom2tpose # 1 x (J+1) x 4 x 4

        # if 'user-defined zero-pose' is entirely 0, use t-pose-template
        else:
            A_custom2current = A_tpose2current


        ## calculate 'forward' pose-deformation matrices (canon->deformed) for all voxels in grid
        t_canon2deform = torch.matmul(
            lbs_weight_grid.reshape(1, N_x * N_y * N_z, N_joint), 
            A_custom2current.view(1, A_custom2current.shape[1], 16)
        ).view(1, -1, 4, 4) # (1 x N_points x J) x (1 x J x 16) -> 1 x N_points x 16 -> 1 x N_points x 4 x 4

        ## calculate 'backward' pose-deformation matrices (deformed->canon) by computing inverse of 'forward' matrices
        t_final = torch.inverse(t_canon2deform) if backward else t_canon2deform

        ## apply 'canonical-space grid' to 'current-space grid' transformation
        # homogeneous coordinates of points (to transform)
        points_homo = torch.cat([points, torch.ones_like(points)[...,0:1]], dim=-1) # 1 x N_points x 4
        
        # apply transformation
        points_current = torch.matmul(t_final, points_homo.unsqueeze(dim=-1)) # (1 x N_points x 4 x 4) x (1 x N_points x 4 x 1) -> (1 x N_points x 4 x 1)
        
        # dehomogenization
        points_current = points_current[:,:,:3,0] # 1 x N_points x 3
        current_grid = points_current[0] # N_points x 3

        ## return human-mesh (smpl-mesh) in 'current-space'
        if not return_human:
            current_human_mesh = None
        else:
            # human mesh lbs-weights
            W = self.core.GLOBAL.smpl_info['lbs_weights'].unsqueeze(dim=0).expand([1, -1, -1]) # 1 x V x (1+J)
            
            # number of joints
            num_joints = self.core.GLOBAL.smpl_info['J_regressor'].shape[0] 
            
            # transformations for each-vertex on 'smpl-mesh'
            T = torch.matmul(
                W, 
                A_tpose2current.view(1, num_joints, 16)
            ).view(1, -1, 4, 4) # (1 x V x (J + 1)) x (1 x (J + 1) x 16) -> (1 x V x 16) -> (1 x V x 4 x 4)
            
            # homogeneous coordinates of each-vertex on 'smpl-mesh'
            v_posed_homo = torch.cat([v_posed, torch.ones_like(v_posed)[...,0:1]], dim=2) # 1 x N_vertices x 4

            # apply transformation
            v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
            verts = v_homo[:, :, :3, 0]

            ## current human-mesh as 'geometry-dict'
            current_human_mesh_dict = {
                'vertices': verts.squeeze().detach().cpu().numpy().astype(np.float32), 
                'triangles': self.core.SMPL.v_template_faces_holistic.squeeze().detach().cpu().numpy().astype(np.long),
                'vertex_colors': self.core.SMPL.v_template_color.squeeze().detach().cpu().numpy(),
            }

            ## current human-mesh as 'open3d' 
            current_human_mesh = o3d.geometry.TriangleMesh()
            current_human_mesh.vertices = o3d.utility.Vector3dVector(current_human_mesh_dict['vertices'])
            current_human_mesh.triangles = o3d.utility.Vector3iVector(current_human_mesh_dict['triangles'])
            
            # color open3d objects 
            if current_human_mesh_dict['vertex_colors'].shape == (3,): 
                current_human_mesh.paint_uniform_color(current_human_mesh_dict['vertex_colors'])
            else:
                current_human_mesh.vertex_colors = o3d.utility.Vector3iVector(current_human_mesh_dict['vertex_colors'])
            current_human_mesh.compute_vertex_normals()

        return current_grid, current_human_mesh, J_transformed
    

    ## discretize & render binary mask per 'discretization-thresholds'
    def render_discretized_distribution(
            self,
            cam_world_key,
            cam_R,
            cam_t,
            cam_K,
            pose,
            betas,
            mask_shape,
            downscale_length,
            use_normalized_K,
            thresholds,
            verbose
        ):

        # prepare inputs
        cam_R = to_np_torch_recursive(cam_R, use_torch=self.use_torch, device=self.device)
        cam_t = to_np_torch_recursive(cam_t, use_torch=self.use_torch, device=self.device)
        cam_K = to_np_torch_recursive(cam_K, use_torch=self.use_torch, device=self.device)

        # (downscaled) target-mask shape
        H, W = mask_shape
        downscale_factor = downscale_length / min(H, W)
        target_H = int(H * downscale_factor) if H > W else downscale_length
        target_W = int(W * downscale_factor) if H < W else downscale_length

        ## 'downscale-factor' for match pixel-scale in cam_K 
        cam_K[0,0] *= downscale_factor
        cam_K[1,1] *= downscale_factor
        cam_K[0,2] *= downscale_factor
        cam_K[1,2] *= downscale_factor

        # retrieve shape of grid
        N_x, N_y, N_z = self.core[cam_world_key].indexgrid.shape[1:]

        ## deform 'learned-probability in canonical space' to 'pose-deformed space'
        deformed_probability_grids_total = self.deform_probability_field(
            cam_world_key=cam_world_key, 
            pose=pose,
            betas=betas,
            return_current_human=True,
            deformation_method="backward-deformation",
            only_return_full=True,
            verbose=verbose,
        )
        
        # deformed-human is same for all body-part clusters: variable to save precomputed human
        human_mask_precomputed = None

        # save rendered-masks per 'discretization thresholds'
        rendered_masks_per_thresh = EasyDict()

        # iterate for 'thresholds'
        for threshold in sorted(thresholds):
            ## render object-mask
            # placeholder to save 'rendered-masks'
            rendered_masks = EasyDict()
            
            # apply 'discretization' to probability in 3D
            threeDmask = deformed_probability_grids_total.full_aggregations[cam_world_key]['deformed_prob_grid'] >= threshold # Shape: NxNxN, bool
    
            # voxel-grid
            cam_world_grid = self.core[cam_world_key].canon_grid.clone() # voxel-grid in 'pose-deformed space'. Shape: 3 x N x N x N 

            # voxels above with 'probability above-threshold'
            cam_world_grid_overthresh = cam_world_grid[:,threeDmask].T # N_over, 3
            
            # project 'binary-grid' (in camera-world) into camera
            projected_grid = perspective_projection_grid(
                grid=cam_world_grid_overthresh,
                K=cam_K,
                R=cam_R,
                t=cam_t,
                grid_size=(len(cam_world_grid_overthresh),),
                use_normalized_K=use_normalized_K,
                img_size=(target_H,target_W)
            ) # Shape: 2 x N_over

            # remove indices if 'projected-points' are outside the image (note: dtype turns to torch.long)
            projected_grid, inside_img = self.retain_projection_inside_img(projected_grid, img_size=(target_H,target_W)) # 2 x N_over_insideimg

            # W,H -> H,W
            projected_grid = torch.flip(projected_grid, dims=[0]) # 2 x N_over_insideimg            

            ## create 'object-rendered-mask' by checking binary occupancies
            # generate as a sparse-tensor (regarding 'coalescence')
            pred_mask = torch.sparse_coo_tensor(
                indices=projected_grid, # 2 x N_over_insideimg
                values=torch.ones_like(projected_grid[0]),
                size=(target_H,target_W)
            ).to_dense()
            # convert as long dtype
            pred_mask = (pred_mask > 0).type(torch.long)
            
            ## fill in the results (object mask)
            rendered_masks['object_mask'] = pred_mask.clone().cpu().numpy() # still torch

            ## render human-mask
            # use precomputed if exists
            if human_mask_precomputed is not None:  
                human_mask = human_mask_precomputed.clone()

            # precompute human-mask if not exists
            else:
                # render 'human-mesh' vertices
                deformed_human = deformed_probability_grids_total.full_aggregations[cam_world_key]['deformed_human_mesh']
                deformed_human_verts = torch.tensor(deformed_human['verts'], dtype=torch.float32).to(self.device) # Shape: V x 3

                # project 'human-mesh' vertices (in camera-world) into camera
                projected_grid_human = perspective_projection_grid(
                    grid=deformed_human_verts,
                    K=cam_K,
                    R=cam_R,
                    t=cam_t,
                    grid_size=(len(deformed_human_verts),),
                    use_normalized_K=use_normalized_K,
                    img_size=(target_H,target_W)
                ) # Shape: 2 x V

                # remove indices if 'projected-points' are outside the image (note: dtype turns to torch.long)
                projected_grid_human, inside_img_human = self.retain_projection_inside_img(projected_grid_human, img_size=(target_H,target_W)) # 2 x V_over_insideimg

                # W,H -> H,W
                projected_grid_human = torch.flip(projected_grid_human, dims=[0]) # 2 x V_over_insideimg
                
                ## create 'human-rendered-mask' by checking binary occupancies
                # generate as a sparse-tensor (regarding 'coalescence')
                human_mask = torch.sparse_coo_tensor(
                    indices=projected_grid_human, # 2 x V_over_insideimg
                    values=torch.ones_like(projected_grid_human[0]),
                    size=(target_H,target_W)
                ).to_dense()
                # convert as long dtype
                human_mask = (human_mask > 0).type(torch.long)

                # save as precompute
                human_mask_precomputed = human_mask.clone()

            ## fill in the results (object mask)
            rendered_masks['human_mask'] = human_mask.clone().cpu().numpy()

            # save to rendered_masks_per_thersh
            rendered_masks_per_thresh[f"{threshold:.6f}"] = rendered_masks

        return rendered_masks_per_thresh, (target_H, target_W)


    ## deform probability field in 'canonical space' to 'pose-deformed space'
    def deform_probability_field(self,
            cam_world_key,
            pose,
            betas,
            return_current_human: bool = True,
            deformation_method = "backward-deformation", # default
            only_return_full=False, # only for evaluation
            eps=torch.tensor([1e-7], dtype=torch.float32),
            verbose=False,
        ):
        ## prepare inputs
        pose = to_np_torch_recursive(pose, use_torch=self.use_torch, device=self.device)
        betas = to_np_torch_recursive(betas, use_torch=self.use_torch, device=self.device)
        eps = to_np_torch_recursive(eps, use_torch=self.use_torch, device=self.device)
        pose = pose.squeeze()
        betas = betas.squeeze()
        
        ## placeholder to return computed values
        deformed_probability_grids = EasyDict(dict(full_aggregations=None, aggregations=None))

        ## we don't need to re-compute deformation result (deformed human-mesh / backward lbs-weights)
        deformed_human_mesh_precomputed = {
            (_cam_world_key, _aggr_world_key): None
            for _aggr_world_key in list(self.aggregations.keys()) + [None] # None used for full
            for _cam_world_key in self.aggregations.keys()
        }
        backward_lbs_weights_precomputed = {
            (_cam_world_key, _aggr_world_key): None
            for _aggr_world_key in list(self.aggregations.keys()) + [None] # None used for full
            for _cam_world_key in self.aggregations.keys()
        }

        ## run deformation for 'full_aggregations'
        # placeholder for saving results
        deformed_probability_grids['full_aggregations'] = EasyDict()
        
        # iterate for all 'camera-world' keys
        for _cam_world_key in self.full_aggregations.keys():
            # for 'full_aggregations', aggregation-world and camera-world are equivalent
            _aggr_world_key = _cam_world_key
                       
            # skip if '_cam_world_key' is not equal to 'cam_world_key' (since '_aggr_world_key == _cam_world_key' in this case)
            if _cam_world_key != cam_world_key:
                deformed_probability_grids['full_aggregations'][_cam_world_key] = None
                continue

            # placeholder for saving results
            deformed_probability_grids['full_aggregations'][_cam_world_key] = EasyDict()
                        
            ## 'deformed' human-mesh
            # use precomputed if exists
            if deformed_human_mesh_precomputed[(_cam_world_key, _aggr_world_key)] is not None:
                deformed_human_mesh = deformed_human_mesh_precomputed[(_cam_world_key, _aggr_world_key)].clone()
            
            # precompute 'deformed' human-mesh if not exists
            else:
                # translations before/after 'skinning-deformation'
                translations = dict(
                    smpl2cam_trans = (-1.) * self.core[_cam_world_key].canoncam2smpl_trans, # 1x3 or None
                    aggr2smpl_trans = self.core[_aggr_world_key].canoncam2smpl_trans, # 1x3 or None
                    deformed2world_trans = None,
                )

                # canonical-grid in 'aggregation world'
                aggr_canon_grid = self.core[_aggr_world_key].canon_grid.clone()
                N_x, N_y, N_z = aggr_canon_grid.shape[1:]
                
                ## situation 1: '(canonical) aggregation world' -> '(pose-deformed) aggregation world' (same as camera-world)
                if _aggr_world_key == cam_world_key:
                    # translate grid in 'aggregation world' before deformation (no translation)
                    pre_translation = torch.zeros([3], dtype=torch.float32).to(self.device)
                    canon_grid_pre_transform = aggr_canon_grid + pre_translation[:,None,None,None] # 3 x N x N x N
                    # forward deformation at 'aggregation world': [N_all_voxels x 3] X [(3*J)] => [N_all_voxels x 3] X [V x 3, F x 3]
                    deformed_canon_grid, deformed_human_mesh, post_translation = self.skinning_deformation(
                        cam_world_key,
                        canon_grid_pre_transform.reshape(3,-1).T, 
                        pose,
                        betas,
                        return_human=return_current_human,
                    )
                    
                ## situation 2: 'aggregation world' -> '(canonical) camera world' -> '(pose-deformed) camera world'
                elif _cam_world_key == cam_world_key:
                    # translate grid in 'aggregation world' to 'camera world' before deformation
                    pre_translation = translations['aggr2smpl_trans'] + translations['smpl2cam_trans']
                    canon_grid_pre_transform = aggr_canon_grid + pre_translation[:,None,None,None] # [3 x N x N x N]
                    # forward deformation at 'camera world': [N_all_voxels x 3] X [(3*J)] => [N_all_voxels x 3] X [V x 3, F x 3]
                    deformed_canon_grid, deformed_human_mesh, post_translation = self.skinning_deformation( 
                        cam_world_key,
                        canon_grid_pre_transform.reshape(3,-1).T,
                        pose, 
                        betas,
                        return_human=return_current_human,
                    )
            
                # save precomputed 'deformed' human-mesh
                deformed_human_mesh_precomputed[(_cam_world_key, _aggr_world_key)] = deformed_human_mesh

            # learned probability field in 'canonical world'. Shape: N x N x N
            canon_visual_hull = self.full_aggregations[_cam_world_key].visual_hull.to(self.device)
            canon_full_image_visual_hull = self.full_aggregations[_cam_world_key].full_image_visual_hull.to(self.device)
            canon_probability_field = canon_visual_hull / torch.where(canon_full_image_visual_hull > eps, canon_full_image_visual_hull, eps)

            ## 'backward' lbs weights
            # deformation-methods
            if deformation_method != "backward-deformation":
                assert deformation_method in ['linear', 'negative-log-mixing'], "Unavailable 'deformation-method'"
                assert False, f"Method '{deformation_method}' deprecated..."

            # use precomputed if exists
            if backward_lbs_weights_precomputed[(_cam_world_key, _aggr_world_key)] is not None:
                backward_lbs_weight_grid = backward_lbs_weights_precomputed[(_cam_world_key, _aggr_world_key)]
            
            # precompute 'backward' lbs-weights if not exists
            else:
                # use deformed human-mesh as template
                v_template_deformed = torch.tensor(np.asarray(deformed_human_mesh.vertices), dtype=torch.float32)
                
                # eqiuspaced-grid in camera world
                equispace_grid_cam_world = self.core[cam_world_key].canon_grid.clone() # 3 x N x N x N

                # compute 'backward' lbs-weights
                backward_lbs_weight_grid = precompute_lbs_weights(
                    canon_grid=equispace_grid_cam_world,
                    v_template=v_template_deformed,
                    lbs_weights=self.core[cam_world_key].lbs_weights.clone(),
                    lbs_precompute_settings=self.core[cam_world_key].lbs_precompute_settings,
                    verbose=verbose
                ).to(self.device).type(torch.float32)
        
                # save to precomputed 'backward' lbs weights
                backward_lbs_weights_precomputed[(_cam_world_key, _aggr_world_key)] = backward_lbs_weight_grid

            ## apply 'backward' deformation
            # align before 'backward' deformation
            grid_deformed_world = equispace_grid_cam_world.reshape(3,-1).T - post_translation
            
            # 'backward' skinning. [N_all_voxels x 3] X [(3*J)] => [N_all_voxels x 3] X [V x 3, F x 3]
            canon_grid_pre_transform, _, _ = self.skinning_deformation(
                cam_world_key,
                grid_deformed_world,
                pose, 
                betas,
                return_human=return_current_human,
                backward=True,
                backward_lbs_weight_grid=backward_lbs_weight_grid,
            )

            # align to 'aggregation world'
            aggr_canon_grid = canon_grid_pre_transform.T.reshape(3,N_x,N_y,N_z) - pre_translation[:,None,None,None] # Shape: 3 x N x N x N
            
            # retrieve indices that fall within 'canonical-space' grid when backward-deformation applied
            leftbottom = self.core[_aggr_world_key].center - 0.5*torch.tensor(
                [
                    self.core[_aggr_world_key].length_x,
                    self.core[_aggr_world_key].length_y,
                    self.core[_aggr_world_key].length_z
                ], 
                dtype=torch.float32
            ).to(self.device)
            aggr_canon_index_grid = aggr_canon_grid - leftbottom[:,None,None,None] # Shape: 3 x N x N x N
            aggr_canon_index_grid /= self.core[_aggr_world_key].voxel_size # Index-Scale. Shape: 3 x N x N x N
            closest_equivoxel_indices = aggr_canon_index_grid.type(torch.long)

            # find indices of voxels that are warped to 'inside' of (canonical) aggr-grid
            is_above_zero = (closest_equivoxel_indices >= 0).all(dim=0) # 3 x N x N x N
            is_x_below_max = closest_equivoxel_indices[0] < N_x
            is_y_below_max = closest_equivoxel_indices[1] < N_y
            is_z_below_max = closest_equivoxel_indices[2] < N_z
            is_below_max = torch.stack([is_x_below_max, is_y_below_max, is_z_below_max], dim=0).all(dim=0)
            is_inside_canongrid = torch.logical_and(is_above_zero, is_below_max) # NxNxN, bool
            
            # if voxels are 'backward' warped to outside of 'canonical' grid, fill probability with 0 value
            closest_equivoxel_indices = torch.where(is_inside_canongrid[None], closest_equivoxel_indices, 0)
            
            # if voxels are 'backward' warped to inside of 'canonical' grid, fill probability with queried values
            deformed_prob_grid = canon_probability_field[
                closest_equivoxel_indices[0,:,:,:],
                closest_equivoxel_indices[1,:,:,:],
                closest_equivoxel_indices[2,:,:,:],
            ]
            deformed_prob_grid = deformed_prob_grid * is_inside_canongrid.type(torch.float)

            # save to return
            deformed_probability_grids['full_aggregations'][_cam_world_key]['deformed_prob_grid'] = deformed_prob_grid.clone().squeeze().detach().cpu().numpy() # N x N x N
            deformed_probability_grids['full_aggregations'][_cam_world_key]['deformed_human_mesh'] = {
                "verts": np.asarray(deformed_human_mesh.vertices),
                "faces": np.asarray(deformed_human_mesh.triangles),
                "color": np.asarray(deformed_human_mesh.vertex_colors) if np.asarray(deformed_human_mesh.vertex_colors).shape[0] != 0 else None
            }

        # run Accumulation for All [_cam_world_key -> body_part_name -> _aggr_world_key]
        deformed_probability_grids['aggregations'] = EasyDict()
        for _cam_world_key in self.aggregations.keys():
            # if only return 'full_aggregations', then skip
            if only_return_full:
                continue

            # placeholder for saving results
            deformed_probability_grids['aggregations'][_cam_world_key] = EasyDict()

            # iterate for all 'body parts'
            for body_part_name in self.core[_cam_world_key].body_part_names:
                # placeholder for saving results
                deformed_probability_grids['aggregations'][_cam_world_key][body_part_name] = EasyDict()
                
                # iterate for all 'aggregation-world key' (for given body-part)
                for _aggr_world_key in self.aggregations[_cam_world_key].aggr_world_name_per_body_parts[body_part_name]:

                    # skip if '_cam_world_key'/'_aggr_world_key' is not equal to 'cam_world_key'
                    if _aggr_world_key != cam_world_key and _cam_world_key != cam_world_key:
                        deformed_probability_grids['aggregations'][_cam_world_key][body_part_name][_aggr_world_key] = None
                        continue
                    
                    # placeholder for saving results
                    deformed_probability_grids['aggregations'][_cam_world_key][body_part_name][_aggr_world_key] = EasyDict()
                    

                    ## 'deformed' human-mesh
                    # use precomputed if exists
                    if deformed_human_mesh_precomputed[(_cam_world_key, _aggr_world_key)] is not None:
                        deformed_human_mesh = deformed_human_mesh_precomputed[(_cam_world_key, _aggr_world_key)]

                    # precompute 'deformed' human-mesh if not exists
                    else:
                        # translations before/after 'skinning-deformation'
                        translations = dict(
                            smpl2cam_trans = (-1.) * self.core[_cam_world_key].canoncam2smpl_trans, # 1x3 or None
                            aggr2smpl_trans = self.core[_aggr_world_key].canoncam2smpl_trans, # 1x3 or None
                            deformed2world_trans = None,
                        )

                        # canonical-grid in 'aggregation world'
                        aggr_canon_grid = self.core[_aggr_world_key].canon_grid.clone()
                        N_x, N_y, N_z = aggr_canon_grid.shape[1:]
                        
                        ## situation 1: '(canonical) aggregation world' -> '(pose-deformed) aggregation world' (same as camera-world)
                        if _aggr_world_key == cam_world_key:
                            # translate grid in 'aggregation world' before deformation (no translation)
                            pre_translation = torch.zeros([3], dtype=torch.float32).to(self.device)
                            canon_grid_pre_transform = aggr_canon_grid + pre_translation[:,None,None,None] # 3 x N x N x N
                            # forward deformation at 'aggregation world': [N_all_voxels x 3] X [(3*J)] => [N_all_voxels x 3] X [V x 3, F x 3]
                            _, deformed_human_mesh, _ = self.skinning_deformation(
                                cam_world_key,
                                canon_grid_pre_transform.reshape(3,-1).T, 
                                pose,
                                betas,
                                return_human=return_current_human,
                            )                            

                        ## situation 2: 'aggregation world' -> '(canonical) camera world' -> '(pose-deformed) camera world'        
                        elif _cam_world_key == cam_world_key:
                            # translate grid in 'aggregation world' to 'camera world' before deformation
                            pre_translation = translations['aggr2smpl_trans'] + translations['smpl2cam_trans']
                            canon_grid_pre_transform = aggr_canon_grid + pre_translation[:,None,None,None] # [3 x N x N x N]
                            # forward deformation at 'camera world': [N_all_voxels x 3] X [(3*J)] => [N_all_voxels x 3] X [V x 3, F x 3]
                            deformed_canon_grid, deformed_human_mesh, post_translation = self.skinning_deformation(
                                cam_world_key,
                                canon_grid_pre_transform.reshape(3,-1).T,
                                pose, 
                                betas,
                                return_human=return_current_human,
                            )

                        # save precomputed 'deformed' human-mesh
                        deformed_human_mesh_precomputed[(_cam_world_key, _aggr_world_key)] = deformed_human_mesh

                    # Deform log-(1-probability) field in canonical space and interpolate
                    canon_visual_hull = self.aggregations[_cam_world_key].visual_hull_per_body_parts[body_part_name][_aggr_world_key].to(self.device)
                    canon_full_image_visual_hull = self.aggregations[_cam_world_key].full_image_visual_hull_per_body_parts[body_part_name][_aggr_world_key].to(self.device)
                    # Canonical probability field, NxNxN
                    canon_probability_field = canon_visual_hull / torch.where(canon_full_image_visual_hull > eps, canon_full_image_visual_hull, eps)
                    
                    ## 'backward' lbs weights
                    # deformation-methods
                    if deformation_method != "backward-deformation":
                        assert deformation_method in ['linear', 'negative-log-mixing'], "Unavailable 'deformation-method'"
                        assert False, f"Method '{deformation_method}' deprecated..."

                    # use precomputed if exists
                    if backward_lbs_weights_precomputed[(_cam_world_key, _aggr_world_key)] is not None:
                        backward_lbs_weight_grid = backward_lbs_weights_precomputed[(_cam_world_key, _aggr_world_key)]

                    # precompute 'backward' lbs-weights if not exists
                    else:
                        # use deformed human-mesh as template
                        v_template_deformed = torch.tensor(np.asarray(deformed_human_mesh.vertices), dtype=torch.float32)
                        
                        # eqiuspaced-grid in camera world
                        equispace_grid_cam_world = self.core[cam_world_key].canon_grid.clone() # 3 x N x N x N
                        
                        # compute 'backward' lbs-weights
                        backward_lbs_weight_grid = precompute_lbs_weights(
                            canon_grid=equispace_grid_cam_world,
                            v_template=v_template_deformed,
                            lbs_weights=self.core[cam_world_key].lbs_weights.clone(),
                            **self.core[cam_world_key].lbs_precompute_settings
                        ).to(self.device).type(torch.float32)

                        # save to precomputed 'backward' lbs weights
                        backward_lbs_weights_precomputed[(_cam_world_key, _aggr_world_key)] = backward_lbs_weight_grid
                            
                    ## apply 'backward' deformation
                    # align before 'backward' deformation
                    grid_deformed_world = equispace_grid_cam_world.reshape(3,-1).T - post_translation
                    
                    # 'backward' skinning. [N_all_voxels x 3] X [(3*J)] => [N_all_voxels x 3] X [V x 3, F x 3]
                    canon_grid_pre_transform, _, _ = self.skinning_deformation(
                        cam_world_key,
                        grid_deformed_world,
                        pose, 
                        betas,
                        return_human=return_current_human,
                        backward=True,
                        backward_lbs_weight_grid=backward_lbs_weight_grid,
                    )

                    # align to 'aggregation world'
                    aggr_canon_grid = canon_grid_pre_transform.T.reshape(3,N_x,N_y,N_z) - pre_translation[:,None,None,None] # [3 x N x N x N]
                    
                    # retrieve indices that fall within 'canonical-space' grid when backward-deformation applied
                    leftbottom = self.core[_aggr_world_key].center - 0.5*torch.tensor(
                        [
                            self.core[_aggr_world_key].length_x,
                            self.core[_aggr_world_key].length_y,
                            self.core[_aggr_world_key].length_z
                        ], dtype=torch.float32
                    ).to(self.device)
                    aggr_canon_index_grid = aggr_canon_grid - leftbottom[:,None,None,None] # Shape: 3 x N x N x N
                    aggr_canon_index_grid /= self.core[_aggr_world_key].voxel_size # Index-Scale. Shape: 3 x N x N x N
                    closest_equivoxel_indices = aggr_canon_index_grid.type(torch.long)

                    # find indices of voxels that are warped to 'inside' of (canonical) aggr-grid
                    is_above_zero = (closest_equivoxel_indices >= 0).all(dim=0) # 3 x N x N x N
                    is_x_below_max = closest_equivoxel_indices[0] < N_x
                    is_y_below_max = closest_equivoxel_indices[1] < N_y
                    is_z_below_max = closest_equivoxel_indices[2] < N_z
                    is_below_max = torch.stack([is_x_below_max, is_y_below_max, is_z_below_max], dim=0).all(dim=0)
                    is_inside_canongrid = torch.logical_and(is_above_zero, is_below_max) # NxNxN, bool

                    # if voxels are 'backward' warped to outside of 'canonical' grid, fill probability with 0 value
                    closest_equivoxel_indices = torch.where(
                        is_inside_canongrid[None], closest_equivoxel_indices, 0)
                    
                    # if voxels are 'backward' warped to inside of 'canonical' grid, fill probability with queried values
                    deformed_prob_grid = canon_probability_field[
                        closest_equivoxel_indices[0,:,:,:],
                        closest_equivoxel_indices[1,:,:,:],
                        closest_equivoxel_indices[2,:,:,:],
                    ]
                    deformed_prob_grid = deformed_prob_grid * is_inside_canongrid.type(torch.float)

                    ## get the probability
                    deformed_probability_grids['aggregations'][_cam_world_key][body_part_name][_aggr_world_key]['deformed_prob_grid'] = deformed_prob_grid.clone().squeeze().detach().cpu().numpy() # NxNxN
                    deformed_probability_grids['aggregations'][_cam_world_key][body_part_name][_aggr_world_key]['deformed_human_mesh'] = {
                        "verts": np.asarray(deformed_human_mesh.vertices),
                        "faces": np.asarray(deformed_human_mesh.triangles),
                        "color": np.asarray(deformed_human_mesh.vertex_colors) if np.asarray(deformed_human_mesh.vertex_colors).shape[0] != 0 else None
                    }

        # if only returning 'full_aggregations', delete the 'aggregations'
        if only_return_full: del deformed_probability_grids['aggregations']

        return deformed_probability_grids


    ## visualize as open3d geometries
    def visualize_current_o3d_geometries(self, verbose):
        # visualize 'full_aggregations'
        for _cam_world_key in self.full_aggregations.keys():
            # print progress
            if verbose:
                print(f"visualization for 'full_aggregations' in _cam_world_key: {_cam_world_key}\n")
            
            # visualization via open3d
            o3d.visualization.draw_geometries(list(self.full_aggregations[_cam_world_key].o3d_geometries.values()))
        
        # visualize 'aggregations'
        for _cam_world_key in self.aggregations.keys():
            if _cam_world_key == 'GLOBAL':
                continue

            # iterate for 'body parts'
            for body_part_name in self.core[_cam_world_key].body_part_names:
                if verbose:
                    print(f"visualization for 'aggregations': _cam_world_key:{_cam_world_key} / body part: {body_part_name} / _aggr_world_key: {self.aggregations[_cam_world_key].aggr_world_name_per_body_parts[body_part_name]}==========\n")
                
                # placeholder to save geometries
                geometries_cluster = []
                for _aggr_world_key in self.aggregations[_cam_world_key].aggr_world_name_per_body_parts[body_part_name]:
                    # update list of geometries
                    geometries_cluster += list(self.aggregations[_cam_world_key].o3d_geometries_per_body_parts[body_part_name][_aggr_world_key].values())
                    
                # visualization via open3d
                o3d.visualization.draw_geometries(geometries_cluster)


    ## auxiliary geometries for additional visualization
    def add_auxiliary_geometry(
            self, 
            _cam_world_key, 
            body_part_name,
            _aggr_world_key,
            name, 
            geometry, 
            exist_ok=False,
            tag_auxiliary=False
        ):
        assert not exist_ok and not tag_auxiliary, "Other settings not implemented..."
        
        ## add auxiliary geometry for 'aggregations'
        if body_part_name is not None:
            assert _aggr_world_key is not None, "'_aggr_world_key' must be provided when 'body_part_name' is given."

            # if duplicate key exists in set of geometries
            if name in self.aggregations[_cam_world_key].o3d_geometries_per_body_parts[body_part_name][_aggr_world_key].keys():
                # raise error 
                assert exist_ok, f"There already exists \"{name}\" in self.aggregations.{_cam_world_key}.{body_part_name}.{_aggr_world_key}"
                # remove existing key and save new 'auxiliary geometry'
                del self.aggregations[_cam_world_key].o3d_geometries_per_body_parts[body_part_name][_aggr_world_key][name]
            
            # add 'tags' in front of key for 'auxiliary geometry'
            if tag_auxiliary:
                name = f"{_cam_world_key}:{body_part_name}:{_aggr_world_key}:{name}"

            # save to set of geometries
            self.aggregations[_cam_world_key].o3d_geometries_per_body_parts[body_part_name][_aggr_world_key][name] = geometry
            self.aggregations[_cam_world_key].geometries_per_body_parts[body_part_name][_aggr_world_key][name] = { # check 'return_triangle_mesh' function
                "verts": np.asarray(geometry.vertices),
                "faces": np.asarray(geometry.triangles),
                "color": np.asarray(geometry.vertex_colors) if np.asarray(geometry.vertex_colors).shape[0] != 0 else None,
                "translation": None,
            }

        ## add auxiliary geometry for 'full_aggregations'
        else:
            assert _aggr_world_key is None, "'_aggr_world_key' must not be provided when 'body_part_name' is not given."
            
            # if duplicate key exists in set of geometries
            if name in self.full_aggregations[_cam_world_key].o3d_geometries.keys():
                # raise error
                assert exist_ok, f"There already exists \"{name}\" in self.full_aggregations['{_cam_world_key}']"
                # remove existing key and save new 'auxiliary geometry'
                del self.full_aggregations[_cam_world_key].o3d_geometries[name]

            # add 'tags' in front of key for 'auxiliary geometry'
            if tag_auxiliary:
                name = f"auxiliary:{_cam_world_key}:{body_part_name}:{_aggr_world_key}:{name}"

            # save to set of geometries
            self.full_aggregations[_cam_world_key].o3d_geometries[name] = geometry
            self.full_aggregations[_cam_world_key].geometries[name] = { # hint: check 'return_triangle_mesh' function
                "verts": np.asarray(geometry.vertices),
                "faces": np.asarray(geometry.triangles),
                "color": np.asarray(geometry.vertex_colors) if np.asarray(geometry.vertex_colors).shape[0] != 0 else None,
                "translation": None,
            }


    ## generate video for single 'aggregation result'
    def generate_video_single(
            self, 
            _cam_world_key, 
            body_part_name, # if None, 'full_aggregations' is visualized
            _aggr_world_key, # if None, 'full_aggregations' is visualized
            save_pth, 
            offscreen,  
            title,
            eps=1e-7,
            auxiliary_geometries=dict(), 
            normalize_max=False,
            tmp_image_cache = None,
            window_size = 500,
            create_video=False,
            title_size=0.1,
            title_color = (0,0,0),
            alpha = 0.8,
            opacity = 0.3, 
            mesh_representation = 'surface', # 'surface', 'wireframe', 'points', 'mesh', 'fancymesh'
            background_color = (1,1,1),
            outline_color = (0,0,0),
            scalar_field_vmin = 0.05,
            scalar_field_vmax = 1.0,
        ):

        # use mayavi.mlab's pipeline's visualizer as off-screen
        if offscreen: mlab.options.offscreen = offscreen
            
        # raise error if '_cam_world_key' not provided
        assert _cam_world_key is not None
                
        # declare mayavi figure
        fig = mlab.figure(size=(window_size, window_size), bgcolor=background_color)

        # voxel-grid settings
        length_x = self.core[_cam_world_key].length_x
        length_y = self.core[_cam_world_key].length_y
        length_z = self.core[_cam_world_key].length_z
        center = self.core[_cam_world_key].center.cpu().numpy()
        voxel_size = self.core[_cam_world_key].voxel_size
        voxel_resolution_x = int(round(length_x / voxel_size, 0))
        voxel_resolution_y = int(round(length_y / voxel_size, 0))
        voxel_resolution_z = int(round(length_z / voxel_size, 0))

        ## visualize probability grid from 'aggregations' as video
        if body_part_name is not None:
            assert _aggr_world_key is not None, "'_aggr_world_key' must be provided when 'body_part_name' is given."
            
            # calculate 'probability grid'
            visual_hull = self.aggregations[_cam_world_key].visual_hull_per_body_parts[body_part_name][_aggr_world_key].detach().cpu().numpy()
            full_image_visual_hull = self.aggregations[_cam_world_key].full_image_visual_hull_per_body_parts[body_part_name][_aggr_world_key].detach().cpu().numpy()
            prob_visual_hull = visual_hull / np.where(full_image_visual_hull > eps, full_image_visual_hull, eps)
            if normalize_max:
                prob_visual_hull = prob_visual_hull / max(prob_visual_hull.max(), eps)

            # geometries
            geometries = self.aggregations[_cam_world_key].geometries_per_body_parts[body_part_name][_aggr_world_key]

        ## visualize probability grid from 'full_aggregations' as video
        else:
            assert _aggr_world_key is None, "'_aggr_world_key' must not be provided when 'body_part_name' is not given."
        
            # calculate 'probability grid'
            visual_hull = self.full_aggregations[_cam_world_key].visual_hull.detach().cpu().numpy()
            full_image_visual_hull = self.full_aggregations[_cam_world_key].full_image_visual_hull.detach().cpu().numpy()
            prob_visual_hull = visual_hull / np.where(full_image_visual_hull > eps, full_image_visual_hull, eps)
            if normalize_max:
                prob_visual_hull = prob_visual_hull / max(prob_visual_hull.max(), eps)

            # geometries
            geometries = self.full_aggregations[_cam_world_key].geometries

        # create scalar-field volume source for visualization (automatically added to 'mayavi.mlab' pipeline when declared)
        mlab.pipeline.volume(mlab.pipeline.scalar_field(prob_visual_hull), vmin=scalar_field_vmin, vmax=scalar_field_vmax)
        mlab.outline(color=outline_color)
        mlab.orientation_axes()

        # visualize human
        for key in list(geometries.keys()):
            # skip if geometry is not human
            if key[:6] != "posed:" and key != "template":
                continue

            # retrieve 'vertices', 'faces(triangles)', 'vertex colors' of 'human' as 'np.ndarray' on 'cpu'
            posed_human = geometries[key]
            vertices = to_np_torch_recursive(deepcopy(posed_human['verts']), use_torch=False, device='cpu') # V x 3
            triangles = to_np_torch_recursive(deepcopy(posed_human['faces']), use_torch=False, device='cpu') # F x 3
            vertex_colors = to_np_torch_recursive(deepcopy(posed_human['color']), use_torch=False, device='cpu') # V x 3

            # broadcast 'vertex_colors' to match dimensions of 'vertices'
            if vertex_colors.shape == (3,):
                vertex_colors = np.broadcast_to(vertex_colors, (vertices.shape[0], 3))

            # move geometry to origin / normalize vertices / move geometry to 'normalized center'
            vertices -= center
            vertices /= voxel_size
            vertices += 0.5*np.array([[voxel_resolution_x,voxel_resolution_y,voxel_resolution_z]])
            
            # add human-mesh to 'mayavi.mlab' pipeline
            posed_human = mlab.triangular_mesh(
                vertices[:,0], 
                vertices[:,1], 
                vertices[:,2], 
                triangles, 
                representation=mesh_representation,
                opacity=opacity,
                scalars = np.arange(vertices.shape[0])
            )

            # add texture to 'posed human'
            posed_human.module_manager.scalar_lut_manager.lut.table = np.c_[
                (vertex_colors * 255), 
                alpha * 255 * np.ones(vertices.shape[0])
            ].astype(np.uint8)

            # update to 'mayavi.mlab' engine
            posed_human.mlab_source.update()
            posed_human.parent.update()

        # visualize auxiliary geometries (these are meshes)
        for key in list(auxiliary_geometries.keys()):
            # retrieve 'vertices', 'faces(triangles)', 'vertex colors' of 'auxiliary geometry' as 'np.ndarray' on 'cpu'
            vertices = to_np_torch_recursive(deepcopy(auxiliary_geometries[key]['verts']), use_torch=False, device='cpu') # V x 3
            triangles = to_np_torch_recursive(deepcopy(auxiliary_geometries[key]['faces']), use_torch=False, device='cpu') # F x 3
            vertex_colors = to_np_torch_recursive(deepcopy(auxiliary_geometries[key]['color']), use_torch=False, device='cpu') # V x 3

            # broadcast 'vertex colors' to match dimensions of 'vertices' 
            if vertex_colors.shape == (3,): vertex_colors = np.broadcast_to(vertex_colors, (vertices.shape[0], 3))

            # move geometry to origin / normalize vertices / move geometry to 'normalized center'
            vertices -= center
            vertices /= voxel_size
            vertices += 0.5*np.array([[voxel_resolution_x,voxel_resolution_y,voxel_resolution_z]]) 

            # add auxiliary geometry (mesh) to 'mayavi.mlab' pipeline
            auxiliary_mesh = mlab.triangular_mesh(
                vertices[:,0], 
                vertices[:,1], 
                vertices[:,2], 
                triangles, 
                representation=mesh_representation,
                opacity=opacity,
                scalars = np.arange(vertices.shape[0]) # used to assign colors based on 'scalars'
            )

            # add texture to auxiliary geometry
            auxiliary_mesh.module_manager.scalar_lut_manager.lut.table = np.c_[
                (vertex_colors * 255), 
                alpha * 255 * np.ones(vertices.shape[0])
            ].astype(np.uint8)

            # update to 'mayavi.mlab' engine
            auxiliary_mesh.mlab_source.update()
            auxiliary_mesh.parent.update()

        # set title
        mlab.title(title, size=title_size, color=title_color)

        # default 'image-cache'
        if tmp_image_cache is None: tmp_image_cache = save_pth.replace(".mp4", "")
        
        # if 'image-cache' already exists, remove the directory and 're-generate'
        if os.path.exists(tmp_image_cache): shutil.rmtree(tmp_image_cache)
        os.makedirs(tmp_image_cache)

        # render geometries from 'various views' and save to 'image-cache'
        for cam_idx in tqdm(range(len(VIDEO_CAMS))):
            # set camera for render
            camera = fig.scene.camera 
            camera.position = VIDEO_CAMS[cam_idx]['position']
            camera.focal_point = VIDEO_CAMS[cam_idx]['focal_point']
            camera.view_angle = VIDEO_CAMS[cam_idx]['view_angle']
            camera.view_up = VIDEO_CAMS[cam_idx]['view_up']
            camera.clipping_range = VIDEO_CAMS[cam_idx]['clipping_range']
            camera.compute_view_plane_normal()
            # save image
            mlab.savefig(f"{tmp_image_cache}/{cam_idx:03}.png", size=(window_size,window_size))

        # close 'mayavi.mlab' figure
        mlab.close(all=True)

        # create video
        if create_video:
            generate_video_from_imgs(tmp_image_cache)

    ## export current 'aggregation results' as video
    def export_video(
            self,
            save_pth,
            prompt,
            delete_cam = False,
            tmp_image_cache = None,
            window_size_per_cluster = 500,
            title_size = 0.1,
            delete_existing_image_cache = True,
            auxiliary_geometries = dict(),
            normalize_max = False,
            offscreen = True,
            verbose = True,
        ):
        # use mayavi.mlab's pipeline's visualizer as off-screen
        if offscreen: mlab.options.offscreen = offscreen

        # default 'image-cache'
        if tmp_image_cache is None: tmp_image_cache = save_pth.replace(".mp4", "")
        
        # if 'image-cache' already exists, remove the directory and 're-generate'
        if delete_existing_image_cache and os.path.exists(tmp_image_cache): shutil.rmtree(tmp_image_cache)
        os.makedirs(tmp_image_cache, exist_ok=True)

        ## generate images
        # generate images for 'full_aggregations'
        directory_num = 0
        for _cam_world_key in self.full_aggregations.keys(): # iterate for '_cam_world_key'
            # number of images aggregated
            num_aggregated = len(self.full_aggregations[_cam_world_key].aggr_input_history.keys())
            # title template
            title = \
                f"prompt:{prompt}\n" \
                f"num_aggregated:{num_aggregated}\n" \
                f"camera-world: {_cam_world_key}\n" \
                f"body-part: none ('full_aggregations')\n" \
                f"aggregation-world: same as camera world ('full_aggregations')"
            # print title
            if verbose:
                print(title)
            
            ## generate images for single 'aggregation result' in 'full_aggregations'
            self.generate_video_single(
                _cam_world_key=_cam_world_key,
                body_part_name=None,
                _aggr_world_key=None,
                save_pth=save_pth, 
                offscreen=offscreen,
                auxiliary_geometries=auxiliary_geometries, 
                normalize_max=normalize_max, 
                tmp_image_cache=f"{tmp_image_cache}/{directory_num:04}",
                window_size = window_size_per_cluster,
                create_video=False,
                title=title,
                title_size=title_size,
            )

            # increase 'directory number'
            directory_num += 1

        # generate images for 'aggregations'
        for _cam_world_key in self.aggregations.keys(): # iterate for '_cam_world_key'
            for body_part_name in self.core[_cam_world_key].body_part_names: # iterate for 'body part'
                for _aggr_world_key in self.aggregations[_cam_world_key].aggr_input_history_per_body_parts[body_part_name].keys(): # iterate for '_aggr_world_key'
                    # number of images aggregated
                    num_aggregated = len(self.aggregations[_cam_world_key].aggr_input_history_per_body_parts[body_part_name][_aggr_world_key].keys())
                    # title tempalte
                    title = \
                        f"prompt:{prompt}\n" \
                        f"num_aggregated:{num_aggregated}\n" \
                        f"camera-world: {_cam_world_key}\n" \
                        f"body-part:{body_part_name}\n" \
                        f"aggregation-world:{_aggr_world_key}"
                    # print title
                    if verbose:
                        print(title)

                    ## generate images for single 'aggregation result' in 'aggregations'
                    self.generate_video_single(
                        _cam_world_key=_cam_world_key,
                        body_part_name=body_part_name,
                        _aggr_world_key=_aggr_world_key,
                        save_pth=save_pth, 
                        offscreen=offscreen,
                        auxiliary_geometries=auxiliary_geometries, 
                        normalize_max=normalize_max, 
                        tmp_image_cache=f"{tmp_image_cache}/{directory_num:04}",
                        window_size = window_size_per_cluster,
                        create_video=False,
                        title=title,
                        title_size=title_size,
                    )

                    # increase 'directory number'
                    directory_num += 1
            

        ## merge all images 'per full_aggregations/aggregations', 'per body-parts' to single image
        # all image paths
        all_image_pths = glob(f"{tmp_image_cache}/*/*.png")
        all_body_part_indices = sorted(list(set([img_pth.split("/")[-2] for img_pth in all_image_pths])))
        all_image_ids = sorted(list(set([img_pth.split("/")[-1].split(".")[0] for img_pth in all_image_pths])))

        # number-of-images, width, height in 'merged frame'
        num_imgs_per_frame = len(all_body_part_indices)
        width = int(math.ceil(math.sqrt(num_imgs_per_frame)))
        height = int(math.ceil(num_imgs_per_frame / width))

        # merge
        for _id in all_image_ids:
            # placeholder for saving merged-images
            total_img = np.zeros([window_size_per_cluster * height, window_size_per_cluster * width, 3], dtype=np.uint8)
            
            # iterate for body-parts
            for body_part_index in all_body_part_indices:
                # open 'single-aggregation' result
                img_pth = f"{tmp_image_cache}/{body_part_index}/{_id}.png"
                image = cv2.imread(img_pth) # Shape: h, w, 3
                
                # fill in the 'single-aggregation' result into 'merged-image'
                body_part_index = int(body_part_index)
                h = body_part_index // width
                w = body_part_index % width
                total_img[
                    window_size_per_cluster*h:window_size_per_cluster*h+window_size_per_cluster, 
                    window_size_per_cluster*w:window_size_per_cluster*w+window_size_per_cluster
                ] = image
                
            # save merged-image
            cv2.imwrite(filename=f"{tmp_image_cache}/{_id}.png",img=total_img)

        # clean sub-directories (each contains 'single-aggregation' results)
        for body_part_index in all_body_part_indices:
            if os.path.exists(f"{tmp_image_cache}/{body_part_index}"): shutil.rmtree(f"{tmp_image_cache}/{body_part_index}")
            
        ## generate mp4 video file
        generate_video_from_imgs(tmp_image_cache)

    ## export aggregation-results
    def export(self, save_pth=None, save_core=True):
        ## export 'aggregations'
        # copy of 'self.aggregations'
        aggregations = deepcopy(self.aggregations)
        
        # remove open3d objects
        for _cam_world_key in aggregations.keys():
            aggregations[_cam_world_key].o3d_geometries_per_body_parts = None

        ## export 'full_aggregations'
        # copy of 'self.full_aggregations'
        full_aggregations = deepcopy(self.full_aggregations)

        # remove open3d objects
        for _cam_world_key in full_aggregations.keys():
            full_aggregations[_cam_world_key].o3d_geometries = None
        
        # save as 'np.ndarrays'
        aggregations = to_np_torch_recursive(aggregations, use_torch=False, device="cpu")
        full_aggregations = to_np_torch_recursive(full_aggregations, use_torch=False, device="cpu")
        
        ## export core
        core = self.export_core(to_cpu=True, to_numpy=True) if save_core else None 


        ## export as dictionary
        export = EasyDict({
            "aggregations": aggregations,
            "full_aggregations": full_aggregations,
            "core": core,
            "use_torch": self.use_torch,
            "device": self.device,
        })
            
        ## return if wanted
        if save_pth is None: return export
        
        ## save if not
        with open(save_pth, "wb") as handle: pickle.dump(export, handle, protocol=pickle.HIGHEST_PROTOCOL)


    ## export 'core'
    def export_core(self, to_cpu=True, to_numpy=True):
        # copy of 'self.core'
        core = deepcopy(self.core)

        # convert to 'np.ndarray' or 'torch.Tensor' / set device
        device = "cpu" if to_cpu else self.device
        core = to_np_torch_recursive(core, use_torch = not to_numpy, device = device)

        return core


    ## load pretrained
    def load(self, load_file=None, load_pth=None, create_o3d_geometries=False):
        # directly load from opened 'load_file'
        if load_file is not None:
            assert load_pth is None, "If 'load_file' is provided, 'load_pth' must not be provided."
        
        # open the 'save_pth' and load file
        else:
            assert load_pth is not None, "If 'load_file' is not provided, 'load_pth' must be provided."
            with open(load_pth, "rb") as handle: load_file = pickle.load(handle)
        
        # load pretrained
        self.aggregations = load_file['aggregations']
        self.full_aggregations = load_file['full_aggregations']
        self.use_torch = load_file['use_torch']
        self.device = load_file['device']

        # create open3d-geometries from loaded pretrained
        if create_o3d_geometries: self._create_o3d_geometries()

        # convert to 'np.ndarray' or 'torch.Tensor'
        self.aggregations = to_np_torch_recursive(self.aggregations, use_torch=self.use_torch, device=self.device)
        self.full_aggregations = to_np_torch_recursive(self.full_aggregations, use_torch=self.use_torch, device=self.device)

        # load 'core' if exist
        if load_file['core'] is not None:
            # load
            self.core = load_file['core']            
            # convert to 'np.ndarray' or 'torch.Tensor'
            self.core = to_np_torch_recursive(self.core, use_torch=self.use_torch, device=self.device)


    ## create open3d-geometries from 'meta-information' of geometries
    def _create_o3d_geometries(self):
        ## create open3d-geometries for 'full_aggregations'
        # iterate for 'camera world'
        for _cam_world_key in self.full_aggregations.keys():
            # placeholder for saving results
            if self.full_aggregations[_cam_world_key].o3d_geometries is None: self.full_aggregations[_cam_world_key].o3d_geometries = EasyDict()
            
            # iterate for geometries in 'full_aggregations'
            for name in self.full_aggregations[_cam_world_key].geometries.keys():
                # 'meta-information' of geometry
                geo_args = self.full_aggregations[_cam_world_key].geometries[name]

                # generator function for open3d objects
                if name == 'grid': o3d_generator = return_cube
                elif name == 'coordinate_frame': o3d_generator = return_coordinate_frame
                elif name == 'template': o3d_generator = return_triangle_mesh
                elif name[:6] == 'posed:': o3d_generator = return_triangle_mesh
                elif name[:4] == 'cam:': o3d_generator = return_camera_frustum
                else: assert False, f"geometry type for '{name}' not implemented"

                # create open3d geometries
                self.full_aggregations[_cam_world_key].o3d_geometries[name] = o3d_generator(**geo_args)
                
        ## create open3d-geometries for 'aggregations'
        # iterate for 'camera world'
        for _cam_world_key in self.aggregations.keys():
            # placeholder for saving results
            if self.aggregations[_cam_world_key].o3d_geometries_per_body_parts is None: self.aggregations[_cam_world_key].o3d_geometries_per_body_parts = EasyDict()
            
            # iterate for 'body part'
            for body_part_name in self.aggregations[_cam_world_key].geometries_per_body_parts.keys():
                # placeholder for saving results
                self.aggregations[_cam_world_key].o3d_geometries_per_body_parts[body_part_name] = EasyDict()
                
                # itearte for 'aggregation world'
                for _aggr_world_key in self.aggregations[_cam_world_key].geometries_per_body_parts[body_part_name].keys():
                    # placeholder for saving results
                    self.aggregations[_cam_world_key].o3d_geometries_per_body_parts[body_part_name][_aggr_world_key] = EasyDict()
                    
                    # iterate for geometries in 'aggregations'
                    for name in self.aggregations[_cam_world_key].geometries_per_body_parts[body_part_name][_aggr_world_key].keys():
                        # 'meta-information' of geometry
                        geo_args = self.aggregations[_cam_world_key].geometries_per_body_parts[body_part_name][_aggr_world_key][name]
                        
                        # generator function for open3d objects
                        if name == 'grid': o3d_generator = return_cube
                        elif name == 'coordinate_frame': o3d_generator = return_coordinate_frame
                        elif name == 'template': o3d_generator = return_triangle_mesh
                        elif name[:6] == 'posed:': o3d_generator = return_triangle_mesh
                        elif name[:4] == 'cam:': o3d_generator = return_camera_frustum
                        else: assert False, f"geometry type for '{name}' not implemented"

                        # create open3d geometries
                        self.aggregations[_cam_world_key].o3d_geometries_per_body_parts[body_part_name][_aggr_world_key][name] = o3d_generator(**geo_args)