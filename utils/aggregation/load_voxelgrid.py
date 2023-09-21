import numpy as np

from utils.misc import get_3d_indexgrid_ijk


def load_voxelgrid(aggr_core):
    for tag in aggr_core.keys():
        # no need to set voxel-grid for 'GLOBAL'
        if tag == 'GLOBAL':
            continue

        # voxel properties
        voxel_size = aggr_core[tag].voxel_size
        voxel_resolution = aggr_core[tag].voxel_resolution
        voxel_radius = voxel_size / 8
        length_x = length_y = length_z = voxel_size * voxel_resolution
        N_x = N_y = N_z = voxel_resolution
        center = aggr_core[tag].center
        center = center if type(center) == np.ndarray else np.array(center)
        start_point = center - np.array([length_x/2, length_y/2, length_z/2])

        # index-grid. By querying with [:,i,j,k], you can get the index [i,j,k].
        indexgrid = get_3d_indexgrid_ijk(N_x, N_y, N_z)

        # canon-space grid. By querying with [:,i,j,k], you can get the canon-world-coordinate values of that grid index.
        canon_grid = start_point.reshape(3,1,1,1) + voxel_size * indexgrid.astype(np.float32) + voxel_size / 2

        # load these information to 'aggr_core'
        aggr_core[tag].length_x = length_x
        aggr_core[tag].length_y = length_y
        aggr_core[tag].length_z = length_z
        aggr_core[tag].N_x = N_x
        aggr_core[tag].N_y = N_y
        aggr_core[tag].N_z = N_z
        aggr_core[tag].start_point = start_point
        aggr_core[tag].voxel_radius = voxel_radius
        aggr_core[tag].indexgrid = indexgrid
        aggr_core[tag].canon_grid = canon_grid

    return aggr_core