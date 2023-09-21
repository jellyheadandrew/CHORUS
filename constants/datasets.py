import json

# coco paths
COCO_METADATA_PTH = "constants/coco_metadata.pickle"
COCO_THING_CLASSES_PTH="constants/coco_thing_classes.json"
COCO_VAL_ANNOTATION_PTH = './imports/COCO/annotations/instances_val2014.json'
COCO_VAL_IMAGE_DIR = './imports/COCO/images/val2014'

# lvis paths
LVIS_METADATA_PTH="constants/lvis_metadata.pickle"
LVIS_THING_CLASSES_PTH="constants/lvis_thing_classes.json"

# coco-eft paths
COCO_EFT_VAL_PTH = './imports/eft/eft_fit/COCO2014-Val-ver10.json'
EXTENDED_COCO_EFT_VAL_SAVE_PTH = './imports/eft/eft_fit/extended_coco2014_eft_val.pickle'

# coco categories: if it exists, load. if it doesn't exist, create & save.
with open(COCO_THING_CLASSES_PTH, "r") as rf:
    coco_thing_classes = json.load(rf)
COCO_CLASS_ID2NAME = {idx: name for idx, name in enumerate(coco_thing_classes)}
COCO_CLASS_NAME2ID = {v: k for k, v in COCO_CLASS_ID2NAME.items()}

# lvis categories
with open(LVIS_THING_CLASSES_PTH, "r") as rf:
    lvis_thing_classes = json.load(rf)
LVIS_CLASS_ID2NAME = {idx: name for idx, name in enumerate(lvis_thing_classes)}
LVIS_CLASS_NAME2ID = {v: k for k, v in LVIS_CLASS_ID2NAME.items()}

# category exceptions
CATEGORY_EXCEPTIONS = {
    'surfboard-demo': COCO_CLASS_NAME2ID['surfboard']
}