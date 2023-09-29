import numpy as np
from easydict import EasyDict
from copy import deepcopy

from constants.frankmocap import BODY_MOCAP_SMPL_PTH
from constants.aggr_settings.default_aggr_core import DEFAULT_AGGR_SETTINGS

from utils.aggregation.load_smpl_info import load_smpl_info
from utils.aggregation.load_voxelgrid import load_voxelgrid
from utils.aggregation.load_body_parts import load_body_parts
from utils.aggregation.load_lbs_weight import load_lbs_weights
from utils.misc import to_np_torch_recursive


def prepare_aggr_core(verbose, **aggr_settings):
    ## replace default settings with provided 'aggr_settings' if exists 
    core_settings = deepcopy(DEFAULT_AGGR_SETTINGS)
    if 'USE_SMPL' in aggr_settings.keys(): core_settings['USE_SMPL'] = aggr_settings['USE_SMPL']
    if 'USE_STAR_POSE_TEMPLATE' in aggr_settings.keys(): core_settings['GLOBAL']['USE_STAR_POSE_TEMPLATE'] = aggr_settings['USE_STAR_POSE_TEMPLATE']
    if 'ZEROBETA' in aggr_settings.keys(): core_settings['GLOBAL']['ZEROBETA'] = aggr_settings['ZEROBETA']
    if 'LEARNEDK' in aggr_settings.keys(): core_settings['GLOBAL']['LEARNEDK'] = aggr_settings['LEARNEDK']
    if 'SYMMETRIC_AUGMENTATION' in aggr_settings.keys(): core_settings['GLOBAL']['SYMMETRIC_AUGMENTATION'] = aggr_settings['SYMMETRIC_AUGMENTATION']
    if 'GRID_RESOLUTION' in aggr_settings.keys(): core_settings['GRID_RESOLUTION'] = aggr_settings['GRID_RESOLUTION']
    if 'GRID_LENGTH_SMPL' in aggr_settings.keys(): core_settings['SMPL']['GRID_LENGTH'] = aggr_settings['GRID_LENGTH_SMPL']
    if 'BODY_PART_DEFINE_METHOD_SMPL' in aggr_settings.keys(): core_settings['SMPL']['BODY_PART_DEFINE_METHOD'] = aggr_settings['BODY_PART_DEFINE_METHOD_SMPL']
    if 'CAMERA_SAMPLING_SMPL' in aggr_settings.keys(): core_settings['SMPL']['CAMERA_SAMPLING'] = aggr_settings['CAMERA_SAMPLING_SMPL']
    if 'LBS_MERGE_METHOD_SMPL' in aggr_settings.keys(): core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_MERGE_METHOD'] = aggr_settings['LBS_MERGE_METHOD_SMPL']
    if 'LBS_SMOOTHING_METHOD_SMPL' in aggr_settings.keys(): core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_SMOOTHING_METHOD'] = aggr_settings['LBS_SMOOTHING_METHOD_SMPL']
    if 'LBS_NUM_NEIGHBORS_SMPL' in aggr_settings.keys(): core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_NUM_NEIGHBORS'] = aggr_settings['LBS_NUM_NEIGHBORS_SMPL']
    if 'LBS_NUM_SMOOTH_TIMES_SMPL' in aggr_settings.keys(): core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_NUM_SMOOTH_TIMES'] = aggr_settings['LBS_NUM_SMOOTH_TIMES_SMPL']
    if 'LBS_Y_SCALE_SMPL' in aggr_settings.keys(): core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_Y_SCALE'] = aggr_settings['LBS_Y_SCALE_SMPL']
    if 'LBS_DIST_ORIGIN_THRESH_SMPL' in aggr_settings.keys(): core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_DIST_ORIGIN_THRESH'] = aggr_settings['LBS_DIST_ORIGIN_THRESH_SMPL']
    if 'LBS_IDENTITY_CENTER_SMPL' in aggr_settings.keys(): core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_IDENTITY_CENTER'] = aggr_settings['LBS_IDENTITY_CENTER_SMPL']
    if 'LBS_IDENTITY_DIST_THRESHOLD_SMPL' in aggr_settings.keys(): core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_IDENTITY_DIST_THRESHOLD'] = aggr_settings['LBS_IDENTITY_DIST_THRESHOLD_SMPL']

    ## prepare aggregation core
    return _aggr_core(verbose=verbose, core_settings=core_settings)


def _aggr_core(verbose, core_settings):
    # aggr_core
    aggr_core = EasyDict()
    smpl_info = load_smpl_info(
        model_path=BODY_MOCAP_SMPL_PTH,
        use_star_pose_template=core_settings['GLOBAL']['USE_STAR_POSE_TEMPLATE']
    )
    
    # GLOBAL core components
    aggr_core.GLOBAL = EasyDict(dict(
        use_star_pose_template = core_settings['GLOBAL']['USE_STAR_POSE_TEMPLATE'], 
        smpl_info = smpl_info,
        visualize_pose_num = core_settings['GLOBAL']['VISUALIZE_POSE_NUM'], # how many current spaces will we visualize per cluster?
        auxiliary_exist_ok = core_settings['GLOBAL']['AUXILIARY_EXIST_OK'],
        tag_auxiliary = core_settings['GLOBAL']['TAG_AUXILIARY'],
        add_human_for_vis = core_settings['GLOBAL']['ADD_HUMAN_FOR_VIS'],
        use_torch = core_settings['GLOBAL']['USE_TORCH'], # use torch or not
        device = core_settings['GLOBAL']['DEVICE'],
        interaction_region_precompute_settings = { # for precomputing interaction region
            'part_interaction_region_precompute_device': core_settings['GLOBAL']['INTERACTION_REGIONS_PRECOMPUTE_SETTINGS']['PART_INTERACTION_REGION_PRECOMPUTE_DEVICE'],
            'part_interaction_region_precompute_chunk_size': core_settings['GLOBAL']['INTERACTION_REGIONS_PRECOMPUTE_SETTINGS']['PART_INTERACTION_REGION_PRECOMPUTE_CHUNK_SIZE'],
        },
        zerobeta = core_settings['GLOBAL']['ZEROBETA'], # which inputs to use
        learned_k = core_settings['GLOBAL']['LEARNED_K'],
        symmetric_augmentation=core_settings['GLOBAL']['SYMMETRIC_AUGMENTATION'], # whether to use symmetric augmentation or not
        max_cam_save_num = core_settings['GLOBAL']['MAX_CAM_SAVE_NUM'],
    ))
    
    # SMPL core components (critical for all grids that learn in SMPL-canonical space)
    aggr_core.SMPL = EasyDict(dict(
        voxel_size = core_settings['SMPL']['GRID_LENGTH']/core_settings['GRID_RESOLUTION'],
        voxel_resolution = core_settings['GRID_RESOLUTION'],
        center = core_settings['SMPL']['CENTER'],
        o3d_voxel_color_filled = core_settings['SMPL']['O3D_VOXEL_COLOR_FILLED'],
        o3d_voxel_color_unfilled = core_settings['SMPL']['O3D_VOXEL_COLOR_UNFILLED'],
        o3d_voxel_sphere_resolution = core_settings['SMPL']['O3D_VOXEL_SPHERE_RESOLUTION'],
        grid_color = core_settings['SMPL']['GRID_COLOR'],
        frustum_size = core_settings['SMPL']['FRUSTUM_SIZE'],
        camera_sampling = core_settings['SMPL']['CAMERA_SAMPLING'], # camera sampling/weighting methods
        use_normalized_K = core_settings['SMPL']['USE_NORMALIZED_K'],
        manual_selection = core_settings['SMPL']['MANUAL_SELECTION'],
        cam_color = core_settings['SMPL']['CAM_COLOR'],
        canoncam2smpl_trans = core_settings['SMPL']['CANONCAM2SMPL_TRANS'], # for smpl, canonical world is smpl-template world itself
        v_template_holistic = smpl_info['v_template'].clone().squeeze().cpu().numpy(), # template mesh vertices of smpl (1x6980x3)
        v_template_faces_holistic = smpl_info['faces'].clone().cpu().numpy(),
        v_template_local = smpl_info['v_template'].clone().squeeze().cpu().numpy(), # template mesh vertices of SMPL (1x6980x3)
        v_template_faces_local = smpl_info['faces'].clone().cpu().numpy(),
        v_template_color = core_settings['SMPL']['V_TEMPLATE_COLOR'], # silver
        body_part_define_method = core_settings['SMPL']['BODY_PART_DEFINE_METHOD'], # clustering method
        body_segmentation_json_pth = core_settings['SMPL']['BODY_SEGMENTATION_JSON_PTH'], # meshcapade segmentation pth
        lbs_weights=smpl_info['lbs_weights'],
        lbs_precompute_settings = dict( # lbs-precompute settings
            precompute_device = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['PRECOMPUTE_DEVICE'],
            chunk_size = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['CHUNK_SIZE'],
            lbs_num_neighbors = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_NUM_NEIGHBORS'],
            lbs_merge_method = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_MERGE_METHOD'],
            lbs_smoothing_method = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_SMOOTHING_METHOD'], 
            lbs_num_smoothing_times = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_NUM_SMOOTH_TIMES'],
            lbs_y_scale = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_Y_SCALE'],
            lbs_dist_origin_thresh = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_DIST_ORIGIN_THRESH'],
            lbs_identity_center = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_IDENTITY_CENTER'],
            lbs_identity_dist_threshold = core_settings['SMPL']['LBS_PRECOMPUTE_SETTINGS']['LBS_IDENTITY_DIST_THRESHOLD'],
        ),
        length_x = None, # voxel_size * N
        length_y = None, # voxel_size * N
        length_z = None, # voxel_size * N
        N_x = None, # N
        N_y = None, # N
        N_z = None, # N
        start_point = None, # -length_x/2, -length_y/2, -length_z/2
        voxel_radius = None,
        indexgrid = None, # 3xNxNxN, torch.int64
        canon_grid = None, # 3xNxNxN, torch.float64
        body_part_names = None,
        body_parts = None,
        epsilon_per_body_part = None, 
        aggr_world_name_per_body_parts = None,
        inter_regions = None,
        lbs_weight_grid = None,
    ))

    # disable core components if you want to
    if not core_settings['USE_SMPL']: aggr_core['SMPL'] = None; del aggr_core['SMPL']; assert 'SMPL' not in aggr_core.keys()

    # fill in missing core-components
    aggr_core = load_voxelgrid(aggr_core) # load voxelgrid
    aggr_core = to_np_torch_recursive(
        aggr_core,
        use_torch=aggr_core.GLOBAL.use_torch,
        device=aggr_core.GLOBAL.device
    )
    aggr_core = load_body_parts(aggr_core, verbose=verbose) # load body parts for semantic clustering
    aggr_core = load_lbs_weights(aggr_core, verbose=verbose) # load precomputed lbs weights
    aggr_core = to_np_torch_recursive(
        aggr_core,
        use_torch=aggr_core.GLOBAL.use_torch,
        device=aggr_core.GLOBAL.device
    )

    return aggr_core