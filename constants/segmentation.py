DETECTRON2_CODE_DIR="./imports/detectron2_repo"

COCO_SEG_CONFIG_PTH="./imports/detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
COCO_SEG_WEIGHTS_PTH="detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"

LVIS_SEG_CONFIG_PTH="LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"
LVIS_SEG_WEIGHTS_PTH="LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"

DEFAULT_SEGMENTATION_THRESHOLD=0.8