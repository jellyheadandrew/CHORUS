DEFAULT_MMPOSE_DETECTOR_KEY = "HRNet+Dark384x288_COCO"
DEFAULT_KEYPOINT_THRESHOLD = 0.7
DEFAULT_KEYPOINT_BBOX_THRESHOLD = 0.0
MMPOSE_DETECTOR_SETTINGS = {
    # HRNet+Dark:384x288, COCO-keypoint
    "HRNet+Dark384x288_COCO":
        [
            f'./imports/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_dark.py',
            'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth',
        ]
}

# keypoint names (coco)
COCO_2DKEYPOINT_IDX2NAME = {
    i: name for i, name in 
    enumerate([
        "nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
        "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
        "left_hip_extra","right_hip_extra","left_knee","right_knee","left_ankle","right_ankle"
        ])
    }
COCO_2DKEYPOINT_NAME2IDX = {v: k for k, v in COCO_2DKEYPOINT_IDX2NAME.items()}