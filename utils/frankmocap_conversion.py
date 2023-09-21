import numpy as np
import cv2

from constants.frankmocap import HMR_INPUT_SIZE


""" reference: https://github.com/facebookresearch/frankmocap """
## convert from 'bounding-box' world to 'original-image' world
def convert_bbox_to_oriIm(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
    imgSize = np.array([imgSizeW,imgSizeH])
    data3D /= boxScale_o2n
    data3D[:,:2] += bboxTopLeft - imgSize * 0.5 + (HMR_INPUT_SIZE * 0.5) / boxScale_o2n
    return data3D


""" reference: https://github.com/facebookresearch/frankmocap """
## convert 'weak-perspective' 3d-joints to 2d-joints
def convert_3djoint_to_2djoint(smpl_joints_3d_vis, imgshape):

    smpl_joints_2d_vis = smpl_joints_3d_vis[:,:2]       # 3D is in camera comaera coordinate with origin on the image center
    smpl_joints_2d_vis[:,0] += imgshape[1]*0.5          # offset to move the origin on the top left
    smpl_joints_2d_vis[:,1] += imgshape[0]*0.5          # offset to move the origin on the top left

    return smpl_joints_2d_vis


""" reference: https://github.com/facebookresearch/frankmocap """
## convert 'smpl' world to 'bounding-box' world
def convert_smpl_to_bbox(data3D, scale, trans, bAppTransFirst=False):
    

    if bAppTransFirst:              # for hand model
        data3D[:,0:2] += trans      # apply translation
        data3D *= scale             # apply scaling
    else:
        data3D *= scale             # apply scaling
        data3D[:,0:2] += trans      # apply translation
    
    data3D *= (HMR_INPUT_SIZE * 0.5)         # 112 is originated from hmr's input size (224,224)

    return data3D


""" reference: https://github.com/facebookresearch/frankmocap """
## generate transformation-matrix
def get_transform(center, scale, res, rot=0):
    h = 200 * scale     # h becomes the original bbox max(height, min). 
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h         # this becomes a scaling factor to rescale original bbox -> res size (default: 224x224)
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


""" reference: https://github.com/facebookresearch/frankmocap """
## transform pixel-location to different frame
def transform(pt, center, scale, res, invert=0, rot=0):
    # generate transformation-matrix
    t = get_transform(center, scale, res, rot=rot)
    
    # to invert-or-not
    if invert:
        t = np.linalg.inv(t)

    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1


""" reference: https://github.com/facebookresearch/frankmocap """
## crop image with given bounding-box
def crop_bboxInfo(img, center, scale, res =(HMR_INPUT_SIZE,HMR_INPUT_SIZE)):
    # upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # bottom right point
    br = np.array(transform([res[0]+1,
                             res[1]+1], center, scale, res, invert=1))-1


    ## new shape (bbox)
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    if new_shape[0] <1  or new_shape[1] <1:
        return None, None, None
    new_img = np.zeros(new_shape, dtype=np.uint8)

    if new_img.shape[0] ==0:
        return None, None, None

    #Compute bbox for legacy format
    bboxScale_o2n = res[0] / new_img.shape[0]             #224 / 531 (legacy: 531)

    # range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    if new_y[0] <0 or new_y[1]<0 or new_x[0] <0 or new_x[1]<0 :
        return None, None, None

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    # bbox-top-left in original = (old_x[0], old_y[0] )
    bboxTopLeft_inOriginal = (ul[0], ul[1] )

    if new_img.shape[0] <20 or new_img.shape[1]<20:
        return None, None, None

    new_img = cv2.resize(new_img, res)

    return new_img, bboxScale_o2n, np.array(bboxTopLeft_inOriginal)


def convert_smpl_to_img(smpl_3d, img, bbox_center, bbox_scale, cam_param_scale, cam_param_trans):
    #Crop image using cropping information
    _, boxScale_o2n, bboxTopLeft = crop_bboxInfo(img, bbox_center, bbox_scale, (HMR_INPUT_SIZE, HMR_INPUT_SIZE) )
    # from bbox space -> img space
    smpl_3d_vis = smpl_3d.copy()
    smpl_3d_vis = convert_smpl_to_bbox(smpl_3d_vis, cam_param_scale, cam_param_trans)
    smpl_3d_vis = convert_bbox_to_oriIm(smpl_3d_vis, boxScale_o2n, bboxTopLeft, img.shape[1], img.shape[0])
    smpl_2d = convert_3djoint_to_2djoint(smpl_3d_vis, img.shape) # 49x2

    return smpl_2d