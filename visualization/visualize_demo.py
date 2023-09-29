import os
import shutil

import torch
import numpy as np
import math
import cv2

from mayavi import mlab

from constants.mayavi_cams import QUAL_CAMS

from utils.misc import to_np_torch_recursive
from utils.transformations import get_focal_length, convert_Kparams_Kmat



def mayavi_render_mesh_heatmap_canonical(
        img_size,
        alpha,
        prob_field,
        verts,
        faces,
        color,
        thres,
        voxel_size,
        fovy,
        interactive,
        ignore_percentage,
        mesh_representation='surface',
        mesh_color=None,
        tmp_cache=None,
    ):
    # set 'mayavi.mlab' as offscreen mode
    mlab.options.offscreen = not interactive
    
    # get image shape
    H, W = img_size
    N_x, N_y, N_z = prob_field.shape

    # declare mayavi figure
    fig = mlab.figure(size=(W,H), bgcolor=(1,1,1))
    scene = fig.scene


    ## Render a 'probability field'
    # prepare 'probability field'
    prob_field = to_np_torch_recursive(prob_field, use_torch=False, device='cpu')
    prob_field = prob_field * (prob_field > ignore_percentage * prob_field.max())
    
    # add to 'mayavi.mlab' pipeline
    mlab.pipeline.volume(mlab.pipeline.scalar_field(prob_field), vmin=thres, vmax=1.0)

    ## Render a 'mesh'
    # prepare 'mesh'
    x = to_np_torch_recursive(verts[:,0], use_torch=False, device='cpu')
    y = to_np_torch_recursive(verts[:,1], use_torch=False, device='cpu')
    z = to_np_torch_recursive(verts[:,2], use_torch=False, device='cpu')
    faces = to_np_torch_recursive(faces, use_torch=False, device='cpu')
    # color
    color = to_np_torch_recursive(color, use_torch=False, device='cpu')
    if color.ndim == 1: color = np.stack([color] * x.shape[0], axis=0)
    if mesh_color is not None: color = np.ones_like(color) * mesh_color

    # add mesh to 'mayavi.mlab' pipeline
    mesh = mlab.triangular_mesh(
        x / voxel_size + N_x / 2,
        y / voxel_size + N_y / 2,
        z / voxel_size + N_z / 2,
        faces,
        representation = mesh_representation,
        scalars = np.arange(x.size)
    )

    # add texture to 'posed human'
    mesh.module_manager.scalar_lut_manager.lut.table = np.c_[
        (color * 255), 
        alpha * 255 * np.ones(verts.shape[0])
    ].astype(np.uint8)

    # update to 'mayavi.mlab' engine
    mesh.mlab_source.update()
    mesh.parent.update()

    ## Set camera for 'canonical' rendering
    renderings = []
    camera = scene.camera
    # fov
    f = get_focal_length(1.0, fovy)
    view_angle = np.arctan((H/2) / (W*f))
    
    ## iterate for all 'QUAL_CAMS'
    for i in range(2):
        # update camera of visualizer
        camera = scene.camera
        camera.position = ((np.array(QUAL_CAMS[i]['position']) / voxel_size) + np.array([N_x,N_y,N_z]) / 2).tolist()
        camera.focal_point = ((np.array(QUAL_CAMS[i]['focal_point']) / voxel_size) + np.array([N_x,N_y,N_z]) / 2).tolist()
        camera.view_angle = np.rad2deg(view_angle) * 2
        camera.view_up = QUAL_CAMS[i]['view_up']
        camera.clipping_range = [0.1, 1000000.]
        camera.compute_view_plane_normal()
        if interactive: break
        
        ## save the scene into 'np.ndarray'
        # [slight hack] save as image, and then re-open it
        os.makedirs(tmp_cache, exist_ok=True)
        mlab.savefig(f"{tmp_cache}/tmp.png")
        rendering = cv2.imread(f"{tmp_cache}/tmp.png")
        rendering = cv2.resize(rendering[:,:,::-1].copy(), dsize=(W,H))
        renderings.append(rendering)

        # remove after opening
        shutil.rmtree(tmp_cache)

    # if 'interactive', activate mlab viewer
    if interactive:
        mlab.show()
    else:
        ## close all figures
        mlab.close(all=True)
        return renderings
    
    
    

def mayavi_render_mesh_heatmap(
        img_size,
        prob_field,
        voxel_size,
        thres,
        cam_R,
        cam_t,
        cam_K,
        verts,
        faces,
        color,
        alpha,
        ignore_percentage,
        interactive,
        mesh_representation='surface',
        mesh_color=None,
        tmp_cache=None,
        **kwargs
    ):
    # set 'mayavi.mlab' as offscreen mode
    mlab.options.offscreen = not interactive
    
    # get image shape
    H, W = img_size
    N_x, N_y, N_z = prob_field.shape

    # declare mayavi figure
    fig = mlab.figure(size=(W,H), bgcolor=(1,1,1))
    scene = fig.scene

    ## Render a 'probability field'
    # prepare 'probability field'
    prob_field = to_np_torch_recursive(prob_field, use_torch=False, device='cpu')
    prob_field = prob_field * (prob_field > ignore_percentage * prob_field.max())
    
    cam_R = to_np_torch_recursive(cam_R, use_torch=False, device='cpu')
    cam_t = to_np_torch_recursive(cam_t, use_torch=False, device='cpu')
    cam_K = to_np_torch_recursive(cam_K, use_torch=False, device='cpu')

    # add to 'mayavi.mlab' pipeline
    mlab.pipeline.volume(mlab.pipeline.scalar_field(prob_field), vmin=thres, vmax=1.0)

    ## Render a 'mesh'
    # prepare 'mesh'
    x = to_np_torch_recursive(verts[:,0], use_torch=False, device='cpu')
    y = to_np_torch_recursive(verts[:,1], use_torch=False, device='cpu')
    z = to_np_torch_recursive(verts[:,2], use_torch=False, device='cpu')
    faces = to_np_torch_recursive(faces, use_torch=False, device='cpu')
    # color
    color = to_np_torch_recursive(color, use_torch=False, device='cpu')
    if color.ndim == 1: color = np.stack([color] * x.shape[0], axis=0)
    if mesh_color is not None: color = np.ones_like(color) * mesh_color

    # add mesh to 'mayavi.mlab' pipeline
    mesh = mlab.triangular_mesh(
        x / voxel_size + N_x / 2,
        y / voxel_size + N_y / 2,
        z / voxel_size + N_z / 2,
        faces,
        representation = mesh_representation,
        scalars = np.arange(x.size)
    )

    # add texture to 'posed human'
    mesh.module_manager.scalar_lut_manager.lut.table = np.c_[
        (color * 255), 
        alpha * 255 * np.ones(verts.shape[0])
    ].astype(np.uint8)

    # update to 'mayavi.mlab' engine
    mesh.mlab_source.update()
    mesh.parent.update()

    ## Set camera
    # translation
    camera_location = -cam_R.T @ cam_t
    # fov
    view_angle = np.arctan( (H / 2) / cam_K[0][0])
    # focal
    focal_point = camera_location - camera_location[2] * cam_R[2] / cam_R[2,2]
    # scale to 'pixel' scale
    camera_location = (camera_location / voxel_size) + np.array([N_x,N_y,N_z]) / 2
    focal_point = (focal_point / voxel_size) + np.array([N_x,N_y,N_z]) / 2
    # update camera of visualizer
    camera = scene.camera
    camera.position = camera_location
    camera.focal_point = focal_point
    camera.view_angle = np.rad2deg(view_angle) * 2 # 30.0 # view_angle
    camera.view_up = -cam_R[1] # negated due to openGL convention
    camera.clipping_range = [0.01, 100000.]
    camera.compute_view_plane_normal()

    # remove toolbar in the figure window
    mlab.process_ui_events()

    ## save the scene into 'np.ndarray'
    # [slight hack] save as image, and then re-open it
    os.makedirs(tmp_cache, exist_ok=True)
    mlab.savefig(f"{tmp_cache}/tmp.png", size=(W,H))
    rendering = cv2.imread(f"{tmp_cache}/tmp.png")
    rendering = cv2.resize(rendering[:,:,::-1].copy(), dsize=(W,H))

    # remove after opening
    shutil.rmtree(tmp_cache)

    # if 'interactive', activate mlab viewer
    if interactive: mlab.show()

    ## close all figures
    mlab.close(all=True)

    return rendering




def mayavi_render_mesh_heatmap_top(
        img_size,
        prob_field,
        voxel_size,
        thres,
        verts,
        faces,
        color,
        alpha,
        fovy,
        ignore_percentage,
        interactive,
        mesh_representation='surface',
        mesh_color=None,
        tmp_cache=None,
        **kwargs
    ):
    # set 'mayavi.mlab' as offscreen mode
    mlab.options.offscreen = not interactive
    
    # get image shape
    H, W = img_size
    N_x, N_y, N_z = prob_field.shape

    # declare mayavi figure
    fig = mlab.figure(size=(W,H), bgcolor=(1,1,1))
    scene = fig.scene

    ## Render a 'probability field'
    # prepare 'probability field'
    prob_field = to_np_torch_recursive(prob_field, use_torch=False, device='cpu')
    prob_field = prob_field * (prob_field > ignore_percentage * prob_field.max())    
    
    cam_R = np.array(
        [
            [1,0,0],
            [0,-1/np.sqrt(2),1/np.sqrt(2)],
            [0,-1/np.sqrt(2),-1/np.sqrt(2)]
        ], dtype=np.float32
    )
    cam_t = np.array([0,0,5])
    cam_K = to_np_torch_recursive(
        convert_Kparams_Kmat(
            W * (
                get_focal_length(1.0, fovy) * torch.Tensor([1,1,0,0])
                + torch.Tensor([0,0,0.5,0.5 * H / W])
            )
        ), 
        use_torch=False, 
        device='cpu'
    )
                             
    # add to 'mayavi.mlab' pipeline
    mlab.pipeline.volume(mlab.pipeline.scalar_field(prob_field), vmin=thres, vmax=1.0)


    ## Render a 'mesh'
    # prepare 'mesh'
    x = to_np_torch_recursive(verts[:,0], use_torch=False, device='cpu')
    y = to_np_torch_recursive(verts[:,1], use_torch=False, device='cpu')
    z = to_np_torch_recursive(verts[:,2], use_torch=False, device='cpu')
    faces = to_np_torch_recursive(faces, use_torch=False, device='cpu')
    # color
    color = to_np_torch_recursive(color, use_torch=False, device='cpu')
    if color.ndim == 1: color = np.stack([color] * x.shape[0], axis=0)
    if mesh_color is not None: color = np.ones_like(color) * mesh_color

    # add mesh to 'mayavi.mlab' pipeline
    mesh = mlab.triangular_mesh(
        x / voxel_size + N_x / 2,
        y / voxel_size + N_y / 2,
        z / voxel_size + N_z / 2,
        faces,
        representation = mesh_representation,
        scalars = np.arange(x.size)
    )

    # add texture to 'posed human'
    mesh.module_manager.scalar_lut_manager.lut.table = np.c_[
        (color * 255), 
        alpha * 255 * np.ones(verts.shape[0])
    ].astype(np.uint8)

    # update to 'mayavi.mlab' engine
    mesh.mlab_source.update()
    mesh.parent.update()


    ## Set camera
    # translation
    camera_location = -cam_R.T @ cam_t
    # fov
    view_angle = np.arctan( (H / 2) / cam_K[0][0])
    # focal
    focal_point = camera_location - camera_location[2] * cam_R[2] / cam_R[2,2]
    # scale to 'pixel' scale
    camera_location = (camera_location / voxel_size) + np.array([N_x,N_y,N_z]) / 2
    focal_point = (focal_point / voxel_size) + np.array([N_x,N_y,N_z]) / 2
    # update camera of visualizer
    camera = scene.camera
    camera.position = camera_location
    camera.focal_point = focal_point
    camera.view_angle = np.rad2deg(view_angle) * 2 # 30.0 # view_angle
    camera.view_up = -cam_R[1] # negated due to openGL convention
    camera.clipping_range = [0.01, 100000.]
    camera.compute_view_plane_normal()

    # remove toolbar in the figure window
    mlab.process_ui_events()

    ## save the scene into 'np.ndarray'
    # [slight hack] save as image, and then re-open it
    os.makedirs(tmp_cache, exist_ok=True)
    mlab.savefig(f"{tmp_cache}/tmp.png", size=(W,H))
    rendering = cv2.imread(f"{tmp_cache}/tmp.png")
    rendering = cv2.resize(rendering[:,:,::-1].copy(), dsize=(W,H))

    # remove after opening
    shutil.rmtree(tmp_cache)

    # if 'interactive', activate mlab viewer
    if interactive: mlab.show()

    ## close all figures
    mlab.close(all=True)

    return rendering