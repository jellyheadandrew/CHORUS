QUAL_AGGR_SETTINGS = {
    "qual:demo":
    dict(
        USE_SMPL = True,
        USE_STAR_POSE_TEMPLATE = True,
        SYMMETRIC_AUGMENTATION = False,
        GRID_LENGTH_SMPL = 3.0,
        GRID_RESOLUTION = 48,
        BODY_PART_DEFINE_METHOD_SMPL = "qual:demo",
        CAMERA_SAMPLING_SMPL = "azimuth-uniform-fine",
        LBS_MERGE_METHOD_SMPL =  "nn_inv_distance_mean+identity_mixing_linear_ratio_with_constant",
        LBS_SMOOTHING_METHOD_SMPL = "laplace-smoothing+identity_thresh_preserve",
        LBS_NUM_NEIGHBORS_SMPL = 30,
        LBS_NUM_SMOOTH_TIMES_SMPL = 30, 
        LBS_Y_SCALE_SMPL = 0.5,
        LBS_DIST_ORIGIN_THRESH_SMPL = 0.8,
        LBS_IDENTITY_CENTER_SMPL = None,
        LBS_IDENTITY_DIST_THRESHOLD_SMPL = 0.8,
        DATA_RATIO = 1.,
        NAME="qual:demo",
    ),
}