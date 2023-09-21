FRANKMOCAP_DIR="./imports/frankmocap"
BODY_MOCAP_REGRESSOR_CKPT=f"{FRANKMOCAP_DIR}/extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt"
BODY_MOCAP_SMPL_DIR=f"{FRANKMOCAP_DIR}/extra_data/smpl"
BODY_MOCAP_SMPL_PTH = f"{BODY_MOCAP_SMPL_DIR}/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
VISUALIZER_MODE="opengl"
HMR_INPUT_SIZE = 224
DEFAULT_EXCONF_FRANKMOCAP = 0.98
DEFAULT_MINOVERLAP_FRANKMOCAP = 0.8
SMPL_KEYPOINTS_IDX2NAME = {
    0:	'pelvis',
    1:	'left_hip',
    2:	'right_hip',
    3:	'spine1',
    4:	'left_knee',
    5:	'right_knee',
    6:	'spine2',
    7:	'left_ankle',
    8:	'right_ankle',
    9:	'spine3',
    10:	'left_foot',
    11:	'right_foot',
    12:	'neck',
    13:	'left_collar',
    14:	'right_collar',
    15:	'head',
    16:	'left_shoulder',
    17:	'right_shoulder',
    18:	'left_elbow',
    19:	'right_elbow',
    20:	"left_wrist",
    21:	"right_wrist",
    22: "left_hand",
    23: "right_hand",
}