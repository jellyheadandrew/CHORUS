# predict segmentations
python scripts/aggregation/predict_segmentations.py \
--mode 'coco' \
--skip_done \
--categories \
'motorcycle' \
'bench' \
'backpack' \
'handbag' \
'tie' \
'frisbee' \
'skis' \
'snowboard' \
'sports ball' \
'baseball glove' \
'skateboard' \
'surfboard' \
'tennis racket' \
'cell phone'

# predict humans
python scripts/aggregation/predict_humans.py \
--skip_done \
--categories \
'motorcycle' \
'bench' \
'backpack' \
'handbag' \
'tie' \
'frisbee' \
'skis' \
'snowboard' \
'sports ball' \
'baseball glove' \
'skateboard' \
'surfboard' \
'tennis racket' \
'cell phone'

# predict keypoint filters
python scripts/aggregation/predict_keypoints.py \
--skip_done \
--categories \
'motorcycle' \
'bench' \
'backpack' \
'handbag' \
'tie' \
'frisbee' \
'skis' \
'snowboard' \
'sports ball' \
'baseball glove' \
'skateboard' \
'surfboard' \
'tennis racket' \
'cell phone'

# filter dataset
python scripts/aggregation/filter_dataset.py \
--skip_done \
--categories \
'motorcycle' \
'bench' \
'backpack' \
'handbag' \
'tie' \
'frisbee' \
'skis' \
'snowboard' \
'sports ball' \
'baseball glove' \
'skateboard' \
'surfboard' \
'tennis racket' \
'cell phone'

# predict perspective camera
python scripts/aggregation/predict_perspective.py \
--skip_done \
--categories \
'motorcycle' \
'bench' \
'backpack' \
'handbag' \
'tie' \
'frisbee' \
'skis' \
'snowboard' \
'sports ball' \
'baseball glove' \
'skateboard' \
'surfboard' \
'tennis racket' \
'cell phone'

# aggregate to 3D
python scripts/aggregation/aggregate.py \
--eval_mode 'quant' \
--aggr_setting_names \
'quant:full' \
--skip_done \
--categories \
'motorcycle' \
'bench' \
'backpack' \
'handbag' \
'tie' \
'frisbee' \
'skis' \
'snowboard' \
'sports ball' \
'baseball glove' \
'skateboard' \
'surfboard' \
'tennis racket' \
'cell phone'