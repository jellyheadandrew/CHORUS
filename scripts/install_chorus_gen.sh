pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r envs/chorus_gen.txt
cd imports/ldm
pip install -e .
cd ../..
pip install -e .