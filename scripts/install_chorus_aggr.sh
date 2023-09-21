pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip install -r envs/chorus_aggr.txt
pip install -U openmim
mim install mmengine
mim install "mmdet==2.27.0" "mmpose==0.29.0" "mmcv-full==1.7.0"
cd imports/frankmocap
pip install -e .
cd ../..
pip install -e .