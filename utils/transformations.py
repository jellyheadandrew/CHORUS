import math
import torch
import torch.nn.functional as F

""" reference: https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py """
def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


""" reference: https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py """
def batch_rigid_transform(
        rot_mats,
        joints,
        parents,
        dtype
    ):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


""" reference: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py """
def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)


""" reference: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py """
def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


""" reference: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py """
def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


""" reference: https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py """
def rotmat_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.

    Conversion equations
    ====================

    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::

        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)

    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = torch.zeros(3, dtype=torch.float32)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = torch.hypot(axis[0], torch.hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = torch.atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta


def rotmat_to_axis_angle_batch(matrix):
    """Convert the rotation matrix into the axis-angle notation.

    Conversion equations
    ====================

    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::

        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)

    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """
    # Batch size.
    batch_size = matrix.shape[0]

    # Axes.
    axis = torch.zeros(batch_size, 3, dtype=torch.float32)
    axis[:,0] = matrix[:,2,1] - matrix[:,1,2]
    axis[:,1] = matrix[:,0,2] - matrix[:,2,0]
    axis[:,2] = matrix[:,1,0] - matrix[:,0,1]

    # Angle.
    r = torch.hypot(axis[:,0], torch.hypot(axis[:,1], axis[:,2]))
    t = matrix[:,0,0] + matrix[:,1,1] + matrix[:,2,2]
    theta = torch.atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r[:,None]

    # Return the data.
    return axis, theta


def convert_Kparams_Kmat(tensor):
    K_elem = torch.split(tensor, 1)
    elem1 = torch.Tensor([1, 0, 0]).to(tensor.device) * K_elem[0]
    elem2 = torch.Tensor([0, 1, 0]).to(tensor.device) * K_elem[1]
    elem3 = torch.Tensor([1, 0, 0]).to(tensor.device) * K_elem[2] + torch.Tensor([0, 1, 0]).to(tensor.device) * K_elem[3] + torch.Tensor([0, 0, 1]).to(tensor.device)

    elem1 = elem1.unsqueeze(0).view(3, -1)
    elem2 = elem2.unsqueeze(0).view(3, -1)
    elem3 = elem3.unsqueeze(0).view(3, -1)

    result = torch.cat((elem1, elem2, elem3), dim=1)
    return result


def batch_convert_Kparams_Kmat(K_elem_batch):
    # K_elem_batch: Bx4
    assert K_elem_batch.ndim == 2
    assert K_elem_batch.shape[-1] == 4
    
    # elem1, elem2, elem3 --> Bx3
    elem1 = torch.Tensor([[1, 0, 0]]).to(K_elem_batch.device) * K_elem_batch[:,0:1]
    elem2 = torch.Tensor([[0, 1, 0]]).to(K_elem_batch.device) * K_elem_batch[:,1:2]
    elem3 = torch.Tensor([[1, 0, 0]]).to(K_elem_batch.device) * K_elem_batch[:,2:3] + \
            torch.Tensor([[0, 1, 0]]).to(K_elem_batch.device) * K_elem_batch[:,3:4] + \
            torch.Tensor([[0, 0, 1]]).to(K_elem_batch.device)

    elem1 = elem1.unsqueeze(1).permute(0,2,1) # Bx3x1
    elem2 = elem2.unsqueeze(1).permute(0,2,1) # Bx3x1
    elem3 = elem3.unsqueeze(1).permute(0,2,1) # Bx3x1

    K = torch.cat((elem1, elem2, elem3), dim=-1)
    return K


def get_focal_length(img_length, fovy, use_radian=False):
    if not use_radian:
        fovy = fovy * math.pi / 180.
    return img_length / (2.0*math.tan(fovy/2))


def get_azimuth(x, z, eps=1e-4, return_rad=False):
    if x >= eps and z >= eps:
        theta = math.atan(z/x)
    elif x < eps and x >= -eps and z >= eps:
        theta = (math.pi / 2)
    elif x < -eps and z >= eps:
        theta = math.pi + math.atan(z/x)
    elif x < -eps and z < eps and z>= - eps:
        theta = math.pi
    elif x < -eps and z < -eps:
        theta = math.pi + math.atan(z/x)
    elif x >= -eps and x < eps and z<-eps:
        theta = 3 * (math.pi / 2)
    elif x >= eps and z<-eps:
        theta = 2 * math.pi + math.atan(z/x)
    else:
        theta = 0.
    if return_rad: return theta
    else: return theta * 180 / math.pi


def get_zenith(x, y, z, eps=1e-4, return_rad=False):
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    cos_zenith = z / r

    if cos_zenith >= 1.-eps:
        zenith = 0.
    elif -1.+eps < cos_zenith < 1.-eps:
        zenith = math.acos(cos_zenith)
    else:
        zenith = math.pi
        
    if return_rad: return zenith
    else: return zenith * 180 / math.pi


def scale_wrt_tan(theta, scale, use_deg):
    # convert theta from 'degree' scale to 'radian' scale
    if use_deg: theta = theta * (2 * math.pi) / 360

    # scale 'tangent' value
    tan = math.tan(theta)
    tan = tan * scale

    # apply 'inverse tangent'
    theta = math.atan(tan)

    # return as 'degree' or 'radian'
    if use_deg: return theta * 360 / (2 * math.pi)
    else: return theta


def perspective_projection_grid(grid, K, R, t, grid_size=None, use_normalized_K=False, img_size=(None, None)):
    # homogeneous coordinates
    homo_grid = torch.matmul(K, torch.matmul(R, grid.T) + t.reshape(3,1)).reshape(3,*grid_size)
    
    # calculate projections
    projected_grid = homo_grid[:2].clone() # Shape: [2,*grid_size]
    projected_grid[0] /= homo_grid[2]
    projected_grid[1] /= homo_grid[2]

    # if using 'normalized-K', upscale to pixel-scale 
    if use_normalized_K:
        projected_grid *= img_size[1] # multiply W

    return projected_grid # [2,*grid_size]


def perspective_projection_mesh(mesh, K, R, t, use_normalized_K=False, img_size=(None, None), return_homo=False):
    # homogeneous coordinates
    homo_mesh = torch.matmul(K, torch.matmul(R, mesh.T) + t.reshape(3,1)).reshape(3,mesh.shape[0])
    
    # calculate projections
    projected_mesh = homo_mesh[:2].clone() # Shape: [2,N]
    projected_mesh[0] /= homo_mesh[2]
    projected_mesh[1] /= homo_mesh[2]

    # if using 'normalized-K', upscale to pixel-scale 
    if use_normalized_K:
        projected_mesh *= img_size[1] # multiply W

    if return_homo: return projected_mesh.T, homo_mesh.T # [N,2], [N,3]
    else: return projected_mesh.T # [N,2]