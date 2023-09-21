# download frankmocap (3d human prediction)
mkdir imports/frankmocap/extra_data
mkdir imports/frankmocap/extra_data/body_module
mkdir imports/frankmocap/extra_data/body_module/pretrained_weights
cd imports/frankmocap/extra_data/body_module
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
mv data data_from_spin
cd pretrained_weights
wget https://dl.fbaipublicfiles.com/eft/2020_05_31-00_50_43-best-51.749683916568756.pt
cd ../../../../..

# download ldm (image generation)
mkdir imports/ldm/models/ldm/stable-diffusion
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -P imports/ldm/models/ldm/stable-diffusion