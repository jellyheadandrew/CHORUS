# Copyright (c) Facebook, Inc. and its affiliates.
"""
Wrapper for Human Pose Estimator using BodyMocap.
See: https://github.com/facebookresearch/frankmocap
"""
import torch

from imports.frankmocap.bodymocap.body_mocap_api import BodyMocap
from imports.frankmocap.renderer.visualizer import Visualizer

from constants.frankmocap import BODY_MOCAP_REGRESSOR_CKPT, BODY_MOCAP_SMPL_PTH, VISUALIZER_MODE


# mocap for body
def prepare_frankmocap_regressor():
    assert torch.cuda.is_available(), "Current version only supports GPU"
    
    # Set mocap regressor
    bodymocap = BodyMocap(
        regressor_checkpoint=BODY_MOCAP_REGRESSOR_CKPT, 
        smpl_dir=None, 
        smplModelPath=BODY_MOCAP_SMPL_PTH
    )
    return bodymocap


def prepare_frankmocap_visualizer():
    assert torch.cuda.is_available(), "Current version only supports GPU"

    # Prepare Pytorch3d Visualizer
    visualizer = Visualizer(VISUALIZER_MODE)
    return visualizer
