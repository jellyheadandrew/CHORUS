import os
from glob import glob

from constants.keypoint_detection import COCO_2DKEYPOINT_NAME2IDX

## printer used for exporting settings
class Printer(object):
    def __init__(self):
        super().__init__()

    def dict2print(self, x: dict, return_str: bool = True):
        lines = []
        for k, v in x.items():
            lines += [f"{k}:{v}"]
        towrite = "\n".join(lines)
        if return_str:
            return towrite
        else:
            print(towrite)

    def dict2txt(self, txt_pth, x: dict):
        lines = []
        for k, v in x.items():
            lines += [f"{k}:{v}"]
        towrite = "\n".join(lines)

        with open(txt_pth, "w") as f:
            f.write(towrite)

    def str2txt(self, txt_pth, x: str):
        with open(txt_pth, "w") as f:
            f.write(x)
PRINTER = Printer()


## create filtering settings
def create_filter_settings(
        ldm_model_key,
        image_dir,
        seg_dir,
        human_dir,
        save_dir,
        keypoint_dir,
        use_template,
        keypoints_which,
        filter_multi_person_object_methods,
        bbox_overlap_thres,
        keypoint_confidence_threshold,
        skip_done,
        verbose,
    ):
    # settings (to save)
    settings = dict()

    ## directory settings
    if not use_template:
        settings["image_dir"] = image_dir
        settings["seg_dir"]=seg_dir
        settings["human_dir"]=human_dir
        settings["keypoint_dir"]=keypoint_dir
    else:
        seg_thres = input("Using Template for Using Directories... Type in the Instance-Segmentation Threshold You Have Used In Previous Steps: ")
        settings["image_dir"] = image_dir
        settings["seg_dir"] = f"{image_dir}_seg_{seg_thres}"
        settings["human_dir"] = f"{image_dir}_frankmocap_seg{seg_thres}"
        settings["keypoint_dir"]=f"{image_dir}_2Dkp_COCO_HRNetDark_frank_seg{seg_thres}"

    ## multi-person, multi-object, bbox-filtering method settings
    # number of person allowed in image
    if "only-one-person-in-image" in filter_multi_person_object_methods:
        assert "multi-person-allowed-in-image" not in filter_multi_person_object_methods, "You must choose between 'only-one-person-in-image' and 'multi-person-allowed-in-image'"
        skip_multi_person_image = True
    elif "multi-person-allowed-in-image" in filter_multi_person_object_methods:
        assert "only-one-person-in-image" not in filter_multi_person_object_methods, "You must choose between 'only-one-person-in-image' and 'multi-person-allowed-in-image'"
        skip_multi_person_image = False
    else:
        assert False, "You must choose between 'only-one-person-in-image' and 'multi-person-allowed-in-image'"
    # number of objectgs allowed in image
    if "only-one-object-in-image" in filter_multi_person_object_methods:
        assert "multi-object-allowed-in-image" not in filter_multi_person_object_methods, "You must choose between 'only-one-object-in-image' and 'multi-object-allowed-in-image'"
        skip_multi_object_image = True
    elif "multi-object-allowed-in-image" in filter_multi_person_object_methods:
        assert "only-one-object-in-image" not in filter_multi_person_object_methods, "You must choose between 'only-one-object-in-image' and 'multi-object-allowed-in-image'"
        skip_multi_object_image = False
    else:       
        assert False, "You must choose between 'only-one-object-in-image' and 'multi-object-allowed-in-image'"
    # If only-one-person allowed in image: how will I count objects?
    if "1-bbox-per-person" in filter_multi_person_object_methods:
        assert "0,1-bbox-per-person" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."
        assert "n-bbox-per-person" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."
        assert "no-bbox-filtering" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."        
        allowed_bbox_num_per_human = [1]
    elif "0,1-bbox-per-person" in filter_multi_person_object_methods:
        assert "1-bbox-per-person" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."
        assert "n-bbox-per-person" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."
        assert "no-bbox-filtering" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."        
        allowed_bbox_num_per_human = [0,1]
    elif "n-bbox-per-person" in filter_multi_person_object_methods:
        assert "1-bbox-per-person" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."
        assert "0,1-bbox-per-person" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."
        assert "no-bbox-filtering" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."        
        allowed_bbox_num_per_human = [0,1,2,3,4,5]
    elif "no-bbox-filtering" in filter_multi_person_object_methods:
        assert "1-bbox-per-person" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."
        assert "0,1-bbox-per-person" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."
        assert "n-bbox-per-person" not in filter_multi_person_object_methods, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."        
        allowed_bbox_num_per_human = []
    else:
        assert False, "You must choose between '1-bbox-per-person' or '0,1-bbox-per-person' or 'n-bbox-per-person' or 'no-bbox-filtering'."    
    # save to settings
    settings["filter_multi_person_object_methods"] = sorted(filter_multi_person_object_methods)

    ## keypoint filter (coco)
    keypoints_that_must_exist, keypoints_that_must_not_exist = keypoints_which.replace(" ", "").replace("\n", "").split("|")
    # keypoints that must exist (as list[list])
    if keypoints_that_must_exist == "":
        keypoints_that_must_exist = []
    else:
        keypoints_that_must_exist = sorted(
            [
                sorted([COCO_2DKEYPOINT_NAME2IDX[name.strip()] for name in name_tuples.split(",")]) 
                for name_tuples in keypoints_that_must_exist.split("/")
            ]
        )
    # keypoints that must not exist (as list[list])
    if keypoints_that_must_not_exist == "":
        keypoints_that_must_not_exist = []
    else:
        keypoints_that_must_not_exist = sorted(
            [
                sorted([COCO_2DKEYPOINT_NAME2IDX[name.strip()] for name in name_tuples.split(",")]) 
                for name_tuples in keypoints_that_must_not_exist.split("/")
            ]
        )
    # save to settings
    settings['keypoints_that_must_exist'] = keypoints_that_must_exist
    settings['keypoints_that_must_not_exist'] = keypoints_that_must_not_exist
    settings["keypoint_confidence_threshold"] = keypoint_confidence_threshold

    ## export settings as string (to save)
    settings = dict(sorted(settings.items()))
    if verbose:
        print("\n\n===settings===")
        PRINTER.dict2print(settings, return_str=False)
    # print as txt file        
    settings_str = PRINTER.dict2print(settings, return_str=True)

    ## auxiliary-or-derived settings (not to save)
    settings["ldm_model_key"] = ldm_model_key

    settings["save_dir"] = save_dir
    settings["bbox_overlap_thres"] = bbox_overlap_thres
    
    settings["skip_done"] = skip_done
    settings["verbose"] = verbose

    settings['skip_multi_person_image'] = skip_multi_person_image
    settings['skip_multi_object_image'] = skip_multi_object_image
    settings['allowed_bbox_num_per_human'] = allowed_bbox_num_per_human

    return settings_str, settings


## finding setting number when given the filter settings
def find_filter_setting_num(settings_str, settings): # if current settings does not exist, saves as new setting
    # create saving directory for the first time
    os.makedirs(settings['save_dir'], exist_ok=True)
    
    # find all existing setting numbers
    setting_nums = sorted(
        [
            int(os.path.splitext(os.path.basename(setting_pth))[0].replace(f"settings:", "")) 
            for setting_pth in list(glob(f"{settings['save_dir']}/*.txt"))
            if os.path.splitext(os.path.basename(setting_pth))[0].replace(f"settings:", "").isdigit()
        ]
    )

    # compare setting with existing setting & find corresponding setting number
    setting_num = None
    for tmp in setting_nums:
        setting_pth = f"{settings['save_dir']}/settings:{tmp:03}.txt"
        with open(setting_pth, "r") as f:
            settings_str_ = f.read()
            if settings_str_ == settings_str:
                setting_num = tmp

    # if no same setting is found, save as new setting number (case1)
    if setting_num is None and len(setting_nums) != 0:
        # new setting number
        setting_num = max(setting_nums)+1
        # save as new .txt file
        new_save_name = f"settings:{setting_num:03}"
        txt_pth = f"{settings['save_dir']}/{new_save_name}.txt"
        PRINTER.str2txt(txt_pth, settings_str)
        # print new settings
        if settings['verbose']:
            print(f"New Settings: {setting_num:03}")
    
    # if no same setting is found, save as new setting number (case2)
    elif setting_num is None and len(setting_nums) == 0:
        # new setting number
        setting_num = 1
        # save as new .txt file
        new_save_name = f"settings:{setting_num:03}"
        txt_pth = f"{settings['save_dir']}/{new_save_name}.txt"
        PRINTER.str2txt(txt_pth, settings_str)
        # print new settings
        if settings['verbose']:
            print(f"New Settings: {setting_num:03}")
    
    # if same setting is found, use that setting number
    else:
        # print settings
        if settings['verbose']:
            print(f"Using already saved Settings: {setting_num:03}")
    
    return setting_num


## load saved settings used during filtering when setting number is given and update to args
def load_filter_settings(args, filter_setting_num):
    # load txt file with filtering settings recorded 
    with open(f"{args.filter_dir}/settings:{filter_setting_num:03}.txt", "r") as rf:
        filter_settings = [line.strip() for line in rf.read().strip().split("\n")]
    
    # 'settings to update' as dictionary
    filter_settings = {
        setting_line.split(":")[0].strip(): setting_line.split(":")[1].strip()
        for setting_line in filter_settings
    }
    filter_settings.update({'filter_setting_num': filter_setting_num})


    # update arguments to filter settings
    for k, v in sorted(vars(args).items()):
        filter_settings[k] = v

    return filter_settings


## load saved settings used during filtering when multiple setting number is given and update to args
def load_filter_settings_combination(args, filter_setting_comb, args_to_ignore=[]):
    ## final filter_settings to return
    filter_settings = dict()

    ## iterate for all filter_setting_num
    for filter_setting_num in filter_setting_comb:
        # load txt file with filtering settings recorded 
        with open(f"{args.filter_dir}/settings:{filter_setting_num:03}.txt", "r") as rf:
            _filter_settings = [line.strip() for line in rf.read().strip().split("\n")]
        
        # 'settings to update' as dictionary
        _filter_settings = {
            setting_line.split(":")[0].strip(): setting_line.split(":")[1].strip()
            for setting_line in _filter_settings
        }

        # for all new 'key-value' in _filter_settings, check for duplicate and consistency
        for k, v in _filter_settings.items():
            if k in filter_settings:
                assert filter_settings[k] == v, f"Consistency between filter_settings: '{filter_setting_comb}' not assured!!"
            else:
                filter_settings[k] = v

    # update arguments to filter settings
    for k, v in sorted(vars(args).items()):
        if k not in args_to_ignore: filter_settings[k] = v

    # update 'filter_setting_numbs' used
    filter_settings.update({'filter_setting_comb': filter_setting_comb})

    return filter_settings


## create name for 'aggregation' to use for save 
def aggr_save_name(filter_setting_comb, aggr_setting_name):
    return f"filter({','.join(str(x) for x in filter_setting_comb)})-aggr({aggr_setting_name})"