python scripts/aggregation/predict_segmentations.py \
--skip_done \
--mode 'coco' \
--categories 'surfboard-demo'
python scripts/aggregation/predict_humans.py \
--skip_done \
--categories 'surfboard-demo'
python scripts/aggregation/predict_keypoints.py \
--skip_done \
--categories 'surfboard-demo'
python scripts/aggregation/filter_dataset.py \
--skip_done \
--categories 'surfboard-demo'
python scripts/aggregation/predict_perspective.py \
--skip_done \
--categories 'surfboard-demo'
python scripts/aggregation/aggregate.py \
--skip_done \
--visualize \
--eval_mode 'qual' \
--aggr_setting_names 'qual:demo' \
--categories 'surfboard-demo'

mkdir results_demo
mv './results/aggregations/stable-v1-4/surfboard-demo/A person balances on a surfboard/filter(3)-aggr(qual:demo).mp4' \
'./results_demo/A person balances on a surfboard.mp4'
mv './results/aggregations/stable-v1-4/surfboard-demo/A person carries a surfboard/filter(3)-aggr(qual:demo).mp4' \
'./results_demo/A person carries a surfboard.mp4'
mv './results/aggregations/stable-v1-4/surfboard-demo/A person paddles on a surfboard/filter(3)-aggr(qual:demo).mp4' \
'./results_demo/A person paddles on a surfboard.mp4'
mv './results/aggregations/stable-v1-4/surfboard-demo/total-aggregation/filter(3)-aggr(qual:demo).mp4' \
'./results_demo/total.mp4'