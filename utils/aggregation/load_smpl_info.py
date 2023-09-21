

import pickle5 as pickle

import numpy as np
import torch

from smplx.utils import Struct
from smplx.lbs import lbs

from utils.misc import to_np


def load_smpl_info(
        model_path = None,
        gender = "neutral",
        age = "adult",
        use_star_pose_template = True,
    ):
    # batch size
    batch_size = 1

    # translation of bones
    create_transl = False # by default
    transl = None # by default
    apply_trans = False # by default

    # data-struct containing all information for smpl
    with open(model_path, "rb") as smpl_file:
        data_struct = Struct(**pickle.load(smpl_file, encoding='latin1'))

    # shapedirs
    shapedirs = data_struct.shapedirs; num_betas = 10
    shapedirs = torch.tensor(to_np(shapedirs[:,:,:num_betas]), dtype=torch.float32)

    # J_regressor
    J_regressor = torch.tensor(to_np(data_struct.J_regressor), dtype=torch.float32)

    # Pose blend shape basis: V x 3 x 207 (hence, 6890 x 3 x 207) --> Coefficients of PCA
    num_pose_basis = data_struct.posedirs.shape[-1]
    posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T # 207 x 20670
    posedirs = torch.tensor(to_np(posedirs), dtype=torch.float32)

    # indices of parents for each joints
    parents = torch.tensor(to_np(data_struct.kintree_table[0])).long()
    parents[0] = -1

    # linear blend skinning weights
    lbs_weights = torch.tensor(to_np(data_struct.weights), dtype=torch.float32)

    # auxiliary
    joint_mapper = None
    faces = data_struct.f
    faces = torch.tensor(to_np(faces, dtype=np.int64), dtype=torch.long)
    pose2rot = True

    # template pose (t-pose)
    v_template_T = data_struct.v_template
    v_template_T = torch.tensor(to_np(v_template_T), dtype=torch.float32) # B x V x 3

    # template pose (star-pose)
    if use_star_pose_template:
        zero_pose = torch.zeros([1,72], dtype=torch.float32)
        zero_pose[:,5] = np.pi / 6 # left hip z-axis rotation 30 degrees clockwise
        zero_pose[:,8] = -np.pi / 6 # right hip z-axis rotation 30 degrees counterclockwise
        zero_betas = torch.zeros([1,10], dtype=torch.float32)
        v_template, joints_v_template = lbs(
            betas=zero_betas,
            pose=zero_pose,
            v_template=v_template_T,
            shapedirs=shapedirs,
            posedirs=posedirs,
            J_regressor=J_regressor,
            parents=parents,
            lbs_weights=lbs_weights,
            pose2rot=pose2rot
        )
    else:
        zero_pose = torch.zeros([1,72], dtype=torch.float32)
        zero_betas = torch.zeros([1,10], dtype=torch.float32)
        v_template, joints_v_template = lbs(
            betas=zero_betas,
            pose=zero_pose,
            v_template=v_template_T,
            shapedirs=shapedirs,
            posedirs=posedirs,
            J_regressor=J_regressor,
            parents=parents,
            lbs_weights=lbs_weights,
            pose2rot=pose2rot
        )
    
    # add t-pose joints, too
    _check, joints_v_template_T = lbs(
        betas=torch.zeros_like(zero_betas),
        pose=torch.zeros_like(zero_pose),
        v_template=v_template_T,
        shapedirs=shapedirs,
        posedirs=posedirs,
        J_regressor=J_regressor,
        parents=parents,
        lbs_weights=lbs_weights,
        pose2rot=pose2rot
    )
    assert torch.allclose(v_template_T.unsqueeze(0), _check)
    
    # return smpl constants
    smpl_info = {
        "v_template_T": v_template_T.unsqueeze(dim=0), # 1 x 6980 x 3
        "v_template": v_template, # 1 x 6980 x 3
        "joints_v_template": joints_v_template, # 1 x 24 x 3
        "joints_v_template_T": joints_v_template_T, # 1 x 24 x 3
        "zero_pose": zero_pose, # 1 x 72
        "zero_betas": zero_betas, # 1 x 10
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "J_regressor": J_regressor,
        "parents": parents,
        "lbs_weights": lbs_weights,
        "pose2rot": pose2rot,
        "faces": faces,
        "SMPL2SMPLX_TRANS": np.array([0.012,-0.184,-0.0185]),
        "SMPLX2SMPL_TRANS": -np.array([0.012,-0.184,-0.0185]),
    }

    return smpl_info