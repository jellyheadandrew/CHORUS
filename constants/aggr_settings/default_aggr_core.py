import numpy as np
import torch

DEFAULT_AGGR_SETTINGS = dict(
    USE_SMPL=True,
    GRID_RESOLUTION=48,
    GLOBAL=dict(
        USE_STAR_POSE_TEMPLATE=True,
        VISUALIZE_POSE_NUM=3,
        AUXILIARY_EXIST_OK=False,
        TAG_AUXILIARY=False,
        ADD_HUMAN_FOR_VIS=False,
        USE_TORCH=True,
        DEVICE='cuda',
        INTERACTION_REGIONS_PRECOMPUTE_SETTINGS=dict(
            PART_INTERACTION_REGION_PRECOMPUTE_DEVICE='cpu',
            PART_INTERACTION_REGION_PRECOMPUTE_CHUNK_SIZE=50000,
        ),
        ZEROBETA=True,
        LEARNED_K=True,
        SYMMETRIC_AUGMENTATION=False,
        MAX_CAM_SAVE_NUM=5,
    ),
    SMPL=dict(
        GRID_LENGTH=3.0,
        CENTER=np.array([0,0,0],dtype=np.float32),
        O3D_VOXEL_COLOR_FILLED=[1,0,0],
        O3D_VOXEL_COLOR_UNFILLED=[0.8,0.8,0.8],
        O3D_VOXEL_SPHERE_RESOLUTION=2,
        GRID_COLOR=[0,0,0],
        FRUSTUM_SIZE=0.25,
        CAMERA_SAMPLING='azimuth-uniform-fine',
        USE_NORMALIZED_K=False,
        MANUAL_SELECTION=None,
        CAM_COLOR=[0,0,0],
        CANONCAM2SMPL_TRANS=np.zeros([3], dtype=np.float32),
        V_TEMPLATE_COLOR=np.array([0.5,0.5,0.5]),
        BODY_PART_DEFINE_METHOD='quant',
        BODY_SEGMENTATION_JSON_PTH="./imports/meshcapade_wiki/assets/SMPL_body_segmentation/smpl/smpl_vert_segmentation.json",
        LBS_PRECOMPUTE_SETTINGS=dict(
            PRECOMPUTE_DEVICE="cpu",
            CHUNK_SIZE=50000,
            LBS_NUM_NEIGHBORS=30,
            LBS_MERGE_METHOD='nn_inv_distance_mean+identity_mixing_linear_ratio_with_constant',
            LBS_SMOOTHING_METHOD='laplace-smoothing+identity_thresh_preserve',
            LBS_NUM_SMOOTH_TIMES=30,
            LBS_Y_SCALE=0.5,
            LBS_DIST_ORIGIN_THRESH=0.8,
            LBS_IDENTITY_CENTER=np.zeros([3], dtype=np.float32),
            LBS_IDENTITY_DIST_THRESHOLD=1.0,
        ),
    )
)