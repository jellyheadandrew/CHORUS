# Installation
CHORUS has been developed and tested on Ubuntu 20.04 with an RTX 3090 GPU device. To get started, follow the installation instructions below.

## Environment Setup
### Environment for Dataset Generation
To set up the conda environment for **dataset generation**, use the following commands:
```shell
conda create --name chorus_gen python=3.8 -y
conda activate chorus_gen
sh scripts/install_chorus_gen.sh
```

### Environment for Aggregation (Learning) & Evaluation
To set up the conda environment for **aggregation (learning) & evaluation**, use the following commands:
```shell
conda create --name chorus_aggr python=3.7 -y
conda activate chorus_aggr
sh scripts/install_chorus_aggr.sh
```

## Download Dependencies
### Download SMPL
Due to licensing constraints, you must manually download the SMPL model from the original website. Follow these steps:

1. Visit the <a href="https://smpl.is.tue.mpg.de">SMPL website</a>.
2. Register, verify your account, and log in.
3. Download `SMPL_python_v.1.1.0.zip` and save it to the following path: `imports/frankmocap/SMPL_python_v.1.1.0.zip`.
4. Run the following command to prepare the SMPL model:

    ```shell
    sh scripts/prepare_smpl.sh
    ```

### Download Off-the-shelf Models
CHORUS utilizes various off-the-shelf models, such as <a href="https://github.com/CompVis/stable-diffusion">stable-diffusion</a> for image generation and <a href="https://github.com/facebookresearch/frankmocap">frankmocap</a> for 3D human prediction. To download these necessary off-the-shelf models, execute the following command:

```shell
sh scripts/download_off_the_shelf.sh
```