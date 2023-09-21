
import numpy as np
import torch
import open3d as o3d


# create triangle mesh from torch.Tensor / np.ndarray input
def return_voxels(
        positions, 
        additional_translation=np.array([0,0,0], dtype=np.float32), 
        radius = 0.01,
        color = [1,0,0]
    ):
    if type(positions) == torch.Tensor:
        positions = positions.squeeze().clone().detach().cpu().numpy()
    if type(additional_translation) == torch.Tensor:
        additional_translation = additional_translation.squeeze().clone().detach().cpu().numpy()

    voxels = []
    for position in positions:
        voxel = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=2)
        voxel.paint_uniform_color(color)
        voxel.translate(position+additional_translation)
        voxels.append(voxel)

    return voxels


# create triangle mesh from torch.Tensor / np.ndarray input
def return_triangle_mesh(verts, faces, color = None, translation = None):
    if type(verts) == torch.Tensor:
        verts = verts.clone().detach().cpu().numpy()
    if type(faces) == torch.Tensor:
        faces = faces.clone().detach().cpu().numpy()
    if type(color) == torch.Tensor:
        color = color.clone().detach().cpu().numpy()
    if type(color) == np.ndarray or type(color) == torch.Tensor:
        color = np.squeeze(color)
    if type(translation) == torch.Tensor:
        translation = translation.clone().detach().cpu().numpy()
    if type(translation) == np.ndarray or type(translation) == torch.Tensor:
        translation = np.squeeze(translation)
    verts = np.squeeze(verts)
    faces = np.squeeze(faces)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    if color is not None:
        if color.shape == (3,):
            mesh.paint_uniform_color(color)
        else:
            assert color.shape == verts.shape
            mesh.vertex_colors = o3d.utility.Vector3dVector(color)

    if translation is not None:
        mesh.translate(translation)

    return mesh


# create open3d cube
def return_cube(length_x, length_y, length_z, center, color=[0.1,0.1,0.1]):
    if type(length_x) == torch.Tensor:
        length_x = length_x.clone().detach().cpu().numpy()
        length_x = np.squeeze(length_x)
    if type(length_y) == torch.Tensor:
        length_y = length_y.clone().detach().cpu().numpy()
        length_y = np.squeeze(length_y)
    if type(length_z) == torch.Tensor:
        length_z = length_z.clone().detach().cpu().numpy()
        length_z = np.squeeze(length_z)
    if type(center) == torch.Tensor:
        center = center.clone().detach().cpu().numpy()
        center = np.squeeze(center)
    if type(color) == torch.Tensor:
        color = color.clone().detach().cpu().numpy()
        color = np.squeeze(color)

    points = np.array([[length_x/2+center[0],length_y/2+center[1],length_z/2+center[2]], # 0
                                [length_x/2+center[0],length_y/2+center[1],-length_z/2+center[2]], # 1
                                [length_x/2+center[0],-length_y/2+center[1],length_z/2+center[2]], # 2
                                [length_x/2+center[0],-length_y/2+center[1],-length_z/2+center[2]], # 3
                                [-length_x/2+center[0],length_y/2+center[1],length_z/2+center[2]], # 4
                                [-length_x/2+center[0],length_y/2+center[1],-length_z/2+center[2]], # 5
                                [-length_x/2+center[0],-length_y/2+center[1],length_z/2+center[2]], # 6
                                [-length_x/2+center[0],-length_y/2+center[1],-length_z/2+center[2]]] # 7
                                , dtype=np.float32) # Shape: (8,3)
    lines = [
        [0,1],
        [0,2],
        [0,4],
        [1,3],
        [1,5],
        [2,3],
        [2,6],
        [3,7],
        [4,5],
        [4,6],
        [5,7],
        [6,7]
    ]     
    cube = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    cube.colors = o3d.utility.Vector3dVector([color]*12)
    return cube


# create open3d camera frustum
def return_camera_frustum(frustum_size, focal_length, cam_R, cam_t, color=[0.1,0.1,0.1]):
    points = [
        np.array([0,0,0], dtype=np.float32),
        np.array([0.5 * frustum_size, 0.5 * frustum_size, focal_length * frustum_size], dtype=np.float32),
        np.array([0.5 * frustum_size, -0.5 * frustum_size, focal_length * frustum_size], dtype=np.float32),
        np.array([-0.5 * frustum_size, 0.5 * frustum_size, focal_length * frustum_size], dtype=np.float32),
        np.array([-0.5 * frustum_size, -0.5 * frustum_size, focal_length * frustum_size], dtype=np.float32),
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 4],
    ]

    # make R, t into numpy
    if type(cam_R) != np.ndarray:
        cam_R = np.array(cam_R, dtype=np.float32)
    elif type(cam_R) == torch.Tensor:
        cam_R = cam_R.detach().cpu().numpy()
    if type(cam_t) != np.ndarray:
        cam_t = np.array(cam_t, dtype=np.float32)
    elif type(cam_t) == torch.Tensor:
        cam_t = cam_t.detach().cpu().numpy()
    
    # rotate and translate via camera transformation
    final_points = []
    for point in points:
        final_point = np.matmul(cam_R.T, point - cam_t)
        final_point = final_point.tolist()
        final_points.append(final_point)

    # camera frustum
    camera_frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(final_points), # as float 3d-vector.
        lines=o3d.utility.Vector2iVector(lines) # as int 2d-vector.
    )
    camera_frustum.colors = o3d.utility.Vector3dVector([color]*8)

    return camera_frustum


# create open3d coordinate frame (centered at world origin)
def return_coordinate_frame(size=1.0):
    # --> centered to origin (which is camera center by default)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size) 