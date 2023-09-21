from copy import deepcopy
from typing import Union
from easydict import EasyDict

import numpy as np
import torch

from constants.datasets import COCO_CLASS_NAME2ID, LVIS_CLASS_NAME2ID, CATEGORY_EXCEPTIONS


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_np_torch_recursive(
    X: Union[dict, EasyDict, np.ndarray, torch.Tensor], 
    use_torch=True, 
    device="cuda",
    np_float_type = np.float32,
    np_int_type = np.long,
    torch_float_type = torch.float32,
    torch_int_type = torch.int64,
    ):

    ## recursive approach
    # for dictionaries, run array-to-tensor recursively
    if type(X) == dict or type(X) == EasyDict:
        for key in X.keys():
            if type(X[key]) in [dict, EasyDict, np.ndarray, torch.Tensor]:
                X[key] = to_np_torch_recursive(X[key], use_torch, device)
    # for np.ndarrays, send to torch.Tensor
    elif type(X) == np.ndarray:
        if use_torch:
            X = torch.tensor(X, device=device)
    # for torch.Tensor, set the device only
    elif type(X) == torch.Tensor:
        if use_torch:
            X = X.to(device)
        else:
            X = X.detach().cpu().numpy()

    ## dtype conversion
    if type(X) == torch.Tensor:
        if X.dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
            X = X.type(torch_float_type)
        elif X.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            X = X.type(torch_int_type)
        else:
            pass
    elif type(X) == np.ndarray:
        if X.dtype in [np.float32, np.float16, np.float64]:
            X = X.astype(np_float_type)
        elif X.dtype in [np.int64, np.int32, np.int16]:
            X = X.astype(np_int_type)
        else:
            pass

    return X


def extract_nonvector(X):
    ## recursive approach
    # for dictionaries, extract non-vector (non np.ndarray / non torch.Tensor) recursively
    new_X = deepcopy(X)
    
    # if X is dict
    if type(X) == dict or type(X) == EasyDict:
        for key in X.keys():
            if type(X[key]) in [dict, EasyDict, list]:
                new_X[key] = extract_nonvector(X[key])    
            elif type(X[key]) in [np.ndarray, torch.Tensor]:
                new_X[key] = "REMOVED (np.ndarray/torch.Tensor)"
            else:
                pass

    # if X is list
    if type(X) == list:
        for i, element in enumerate(X):
            if type(element) in [dict, EasyDict, list]:
                new_X[i] = extract_nonvector(element)
            elif type(element) in [np.ndarray, torch.Tensor]:
                new_X[i] = "REMOVED (np.ndarray/torch.Tensor)"
            else:
                pass

    return new_X


def get_3d_indexgrid_ijk(N_x, N_y, N_z, raveled=False):
    # Create indice grid
    indices =np.mgrid[0:N_x,0:N_y,0:N_z]
    if raveled:
        indices = np.stack([indices[0].ravel()+1,
                            indices[1].ravel()+1,
                            indices[2].ravel()+1], axis=-1)
        # -> Shape: [(N_x+1) * (N_y+1) * (N_z+1), 3]
        # -> NOTE: Order would be 
        #            0 0 0
        #            0 0 1
        #            0 0 2
        #            ...
        #            0 1 0
        #            0 1 1
        #            ...
        #            1 0 0
        #            1 0 1
        #            ...
    return indices


def load_category_id(category):
    # load category: coco
    if category in list(COCO_CLASS_NAME2ID.keys()):
        category_id = COCO_CLASS_NAME2ID[category]
    # load category: lvis
    elif category in list(LVIS_CLASS_NAME2ID.keys()):
        category_id = LVIS_CLASS_NAME2ID[category]
    # load category: exceptions (e.g., 'surfboard-demo' -> 'surfboard' id)
    elif category in CATEGORY_EXCEPTIONS.keys():
        category_id = CATEGORY_EXCEPTIONS[category]
    else:
        assert False, f"method not implemented for [category: {category}] yet"

    return category_id    