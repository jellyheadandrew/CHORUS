import numpy as np
import torch


def load_lbs_weights(aggr_core, verbose=False):
    ## iterate for all tags in 'aggr_core'
    for base_tag in aggr_core.keys():
        # no need to set lbs-weights for 'GLOBAL'
        if base_tag == 'GLOBAL':
            continue

        # prepare data
        canon_grid = aggr_core[base_tag].canon_grid
        v_template = aggr_core[base_tag].v_template_holistic
        lbs_weights = aggr_core[base_tag].lbs_weights
        lbs_precompute_settings = aggr_core[base_tag].lbs_precompute_settings
        
        # compute lbs weight for all voxels in grid
        lbs_weight_grid = precompute_lbs_weights(
            canon_grid=canon_grid,
            v_template=v_template,
            lbs_weights=lbs_weights,
            lbs_precompute_settings=lbs_precompute_settings,
            verbose=verbose
        )

        # update to 'aggr_core' with precomputed 'lbs_weight_grid'
        aggr_core[base_tag].lbs_weight_grid = lbs_weight_grid
    
    return aggr_core



def precompute_lbs_weights(
    canon_grid, # 3xNxNxN
    v_template, # Vx3
    lbs_weights, # VxJ
    lbs_precompute_settings,
    verbose=False,
    ):
 
    ## prepare data
    # calculate the distance from grid to all smpl mesh vertices
    N_x, N_y, N_z = canon_grid.shape[1:]; batch_size = 1 

    # ravel from the shape of (3xNxNxN)
    canon_grid = canon_grid.to(lbs_precompute_settings['precompute_device'])
    grid_raveled = canon_grid.reshape(3, -1).T

    # smpl mesh template (Vx3)
    v_template = v_template.to(lbs_precompute_settings['precompute_device'])

    # lbs-weights per vertex (VxJ)
    lbs_weights = lbs_weights.to(lbs_precompute_settings['precompute_device'])


    ## find nearest neighbor vertices
    # calculate distance from all points to all vertices in SMPL
    displacements = grid_raveled[None,:,:] - v_template.squeeze()[:,None,:] # 6980 x N_all_voxels, 3
    
    ## precomputation kills cpu if implemented naively: 'precompute per chunks'
    # placeholder to save distance between 'canon-voxel' and 'body_part vertex'
    distances = torch.empty(displacements.shape[:2], dtype=torch.float32)
    
    # iterate for all 'chunks'
    displacements_done = 0
    while displacements_done < displacements.shape[1]:
        # print progress
        if verbose:
            print("\n===== precomputing lbs-weights for all voxels ... =====")
            print("[lbs-weights] precomputing: ", displacements_done)
    
        # which chunk to precompute
        start_idx = displacements_done
        end_idx = min(displacements_done + lbs_precompute_settings['chunk_size'], displacements.shape[1])

        # compute 'distance between 'body-part vertices' and 'canon-grid voxels'
        distances[:,start_idx:end_idx] = displacements[:,start_idx:end_idx].square().sum(dim=-1).sqrt()

        # update 'displacements done' to move to next chunk
        displacements_done = end_idx
        
    # remove used 'dsplacements' variable & print 'end' for precomputation
    del displacements
    if verbose:
        print("===== precomputing lbs weights: done! =====\n")
    
    ## get top-K closest neighbors (output: N_neighbors x N_all_voxels)
    # top-K neighbors per voxel
    topk_neighbor_distances, topk_neighbor_indices = torch.topk(
        distances, 
        k=lbs_precompute_settings['lbs_num_neighbors'], 
        dim=0, 
        largest = False, 
        sorted=True
    )

    # top-1 neighbor per voxel
    top1_neighbor_distances = topk_neighbor_distances[0] # N_all_voxels


    ## query the nearest vertices skinning weights (N_neighbors x N_all_voxels x J)
    # lbs_weights of 'K' neighbors
    lbs_weights_of_neighbors = lbs_weights[topk_neighbor_indices]
    
    # shape
    N_all_voxels, N_joint = lbs_weights_of_neighbors.shape[1:]
    
    ## calculate skinning weights
    # method 1: identity (no canonicalization/warping)
    if lbs_precompute_settings['lbs_merge_method'] == "identity":
        lbs_weights_per_voxel = torch.zeros(
            [batch_size, N_all_voxels, N_joint], 
            device=lbs_precompute_settings['precompute_device']
        )
        lbs_weights_per_voxel[...,0] = 1 # batch x N_all_voxels x J
        force_identity = None

    # method 2: nearest-neighbor naive mean
    elif lbs_precompute_settings['lbs_merge_method'] == "nn_mean":
        lbs_weights_per_voxel = lbs_weights_of_neighbors.mean(dim=0).expand([batch_size, -1, -1]) # batch x N_all_voxels x J
        force_identity = None

    # method 3: nearest neighbor inverse-distance weighted mean
    elif lbs_precompute_settings['lbs_merge_method'] == "nn_inv_distance_mean":
        exactly_smpl_vertice = (top1_neighbor_distances == 0)[None,:] # 1 x N_all_voxels
        to_replace = torch.zeros_like(topk_neighbor_distances); to_replace[0, :] = 1 # N_neighbors x N_all_voxels
        inverse_neighbor_distances = torch.where(exactly_smpl_vertice, torch.zeros_like(topk_neighbor_distances), 1./topk_neighbor_distances)  # N_neighbors x N_all_voxels
        neighbor_weights = inverse_neighbor_distances / inverse_neighbor_distances.sum(dim=0, keepdim=True).clamp(min=1e-10) # Shape: N_neighbors x N_all_voxels
        assert not torch.any(torch.isnan(neighbor_weights)), "NaN problems"
        neighbor_weights += exactly_smpl_vertice.float() * to_replace # N_neighbors x N_all_voxels
        lbs_weights_per_voxel = (lbs_weights_of_neighbors * neighbor_weights[...,None]).sum(dim=0).expand([batch_size,-1,-1]) # batch x N_all_voxels x J
        force_identity = None

    # method 4: nearest neighbor inverse-distance weighted mean + ratio identity mixing (deweighting)
    elif lbs_precompute_settings['lbs_merge_method'] == "nn_inv_distance_mean+identity_mixing_ratio":
        # inverse distance weighted sum for nearest neighbors
        exactly_smpl_vertice = (topk_neighbor_distances[0] == 0)[None,:] # 1 x N_all_voxels
        to_replace = torch.zeros_like(topk_neighbor_distances); to_replace[0, :] = 1 # N_neighbors x N_all_voxels
        inverse_neighbor_distances = torch.where(exactly_smpl_vertice, torch.zeros_like(topk_neighbor_distances), 1./topk_neighbor_distances)  # N_neighbors x N_all_voxels
        neighbor_weights = inverse_neighbor_distances / inverse_neighbor_distances.sum(dim=0, keepdim=True).clamp(min=1e-10) # Shape: N_neighbors x N_all_voxels
        assert not torch.any(torch.isnan(neighbor_weights)), "NaN problems"
        neighbor_weights += exactly_smpl_vertice.float() * to_replace # N_neighbors x N_all_voxels
        lbs_weights_per_voxel = (lbs_weights_of_neighbors * neighbor_weights[...,None]).sum(dim=0).expand([batch_size,-1,-1]) # batch x N_all_voxels x J
        # identity skinning weights
        identity_skin_weights = torch.zeros([batch_size, N_all_voxels, N_joint], device=lbs_precompute_settings['precompute_device'])
        identity_skin_weights[...,0] = 1 # batch x N_all_voxels x J
        # soft mixing weights for identity and nearest neighbor skinning weights
        dists_from_origin = (grid_raveled - lbs_precompute_settings["lbs_identity_center"][None].to(lbs_precompute_settings['precompute_device'])).square().sum(dim=-1).sqrt() # N_all_voxels
        force_identity = (dists_from_origin > lbs_precompute_settings["lbs_identity_dist_threshold"]).float() # N_all_voxels, 0. or 1.
        ratios = top1_neighbor_distances / dists_from_origin # N_all_voxels. NOTE Closer to 1, bigger the influence of identity
        weights_for_identity = zero2one_to_zero2inf(ratios, method='arctanh', y_scale=lbs_precompute_settings["lbs_y_scale"]) # N_all_voxels
        weights_for_identity = 1 - torch.exp(-weights_for_identity)
        weights_for_identity = weights_for_identity * (1 - force_identity) + torch.ones_like(weights_for_identity) * force_identity
        # soft mixing
        weights_for_identity = weights_for_identity[None,:,None]
        lbs_weights_per_voxel = identity_skin_weights * weights_for_identity + lbs_weights_per_voxel * (1 - weights_for_identity)

    # method 5: nearest neighbor inverse-distance weighted mean + linear identity mixing (deweighting)
    elif lbs_precompute_settings['lbs_merge_method'] == "nn_inv_distance_mean+identity_mixing_linear":
        # inverse distance weighted sum for nearest neighbors
        exactly_smpl_vertice = (topk_neighbor_distances[0] == 0)[None,:] # 1 x N_all_voxels
        to_replace = torch.zeros_like(topk_neighbor_distances); to_replace[0, :] = 1 # N_neighbors x N_all_voxels
        inverse_neighbor_distances = torch.where(exactly_smpl_vertice, torch.zeros_like(topk_neighbor_distances), 1./topk_neighbor_distances)  # N_neighbors x N_all_voxels
        neighbor_weights = inverse_neighbor_distances / inverse_neighbor_distances.sum(dim=0, keepdim=True).clamp(min=1e-10) # Shape: N_neighbors x N_all_voxels
        assert not torch.any(torch.isnan(neighbor_weights)), "NaN problems"
        neighbor_weights += exactly_smpl_vertice.float() * to_replace # N_neighbors x N_all_voxels
        lbs_weights_per_voxel = (lbs_weights_of_neighbors * neighbor_weights[...,None]).sum(dim=0).expand([batch_size,-1,-1]) # batch x N_all_voxels x J
        # identity skinning weights
        identity_skin_weights = torch.zeros([batch_size, N_all_voxels, N_joint], device=lbs_precompute_settings['precompute_device'])
        identity_skin_weights[...,0] = 1 # batch x N_all_voxels x J
        # soft mixing weights for identity and nearest neighbor skinning weights
        dists_from_origin = (grid_raveled - lbs_precompute_settings["lbs_identity_center"][None].to(lbs_precompute_settings['precompute_device'])).square().sum(dim=-1).sqrt() # N_all_voxels
        force_identity = (dists_from_origin > lbs_precompute_settings["lbs_identity_dist_threshold"]).float() # N_all_voxels, 0. or 1.
        identity_distances = lbs_precompute_settings["lbs_dist_origin_thresh"] * torch.ones_like(top1_neighbor_distances) # N_all_voxels
        weights_for_identity = top1_neighbor_distances / (identity_distances + top1_neighbor_distances) # N_all_voxels. NOTE top1_neighbor_distances small means weights for identity small
        weights_for_identity = weights_for_identity * (1 - force_identity) + torch.ones_like(weights_for_identity) * force_identity
        # soft mixing
        weights_for_identity = weights_for_identity[None,:,None] # 1 x N_all_voxels x 1
        lbs_weights_per_voxel = identity_skin_weights * weights_for_identity + lbs_weights_per_voxel * (1 - weights_for_identity) # batch x N_all_voxels x J

    # method 6: nearest neighbor inverse-distance weighted mean + linear & ratio identity mixing (deweighting) --> selected method
    elif lbs_precompute_settings['lbs_merge_method'] == "nn_inv_distance_mean+identity_mixing_linear_ratio_with_constant":
        # inverse distance weighted sum for nearest neighbors
        exactly_smpl_vertice = (topk_neighbor_distances[0] == 0)[None,:] # 1 x N_all_voxels
        to_replace = torch.zeros_like(topk_neighbor_distances); to_replace[0, :] = 1 # N_neighbors x N_all_voxels
        inverse_neighbor_distances = torch.where(exactly_smpl_vertice, torch.zeros_like(topk_neighbor_distances), 1./topk_neighbor_distances)  # N_neighbors x N_all_voxels
        neighbor_weights = inverse_neighbor_distances / inverse_neighbor_distances.sum(dim=0, keepdim=True).clamp(min=1e-10) # Shape: N_neighbors x N_all_voxels
        assert not torch.any(torch.isnan(neighbor_weights)), "NaN problems"
        neighbor_weights += exactly_smpl_vertice.float() * to_replace # N_neighbors x N_all_voxels
        lbs_weights_per_voxel = (lbs_weights_of_neighbors.to(lbs_precompute_settings['precompute_device']) * neighbor_weights[...,None].to(lbs_precompute_settings['precompute_device'])).sum(dim=0).expand([batch_size,-1,-1]) # batch x N_all_voxels x J
        # identity skinning weights
        identity_skin_weights = torch.zeros([batch_size, N_all_voxels, N_joint], device=lbs_precompute_settings['precompute_device'])
        identity_skin_weights[...,0] = 1 # batch x N_all_voxels x J
        # soft mixing weights for identity, NOT via origin. --> set as 0.75
        force_identity = (top1_neighbor_distances > lbs_precompute_settings["lbs_identity_dist_threshold"]).float() # N_all_voxels, 0. or 1.
        force_identity = force_identity.to(lbs_precompute_settings['precompute_device'])
        # identity distance ratio        --> lbs_dist_origin_thresh: set as 0.5
        identity_distances = lbs_precompute_settings["lbs_dist_origin_thresh"] * torch.ones_like(top1_neighbor_distances) # N_all_voxels
        dist_from_body_ratios = top1_neighbor_distances / identity_distances
        # weights for identity
        weights_for_identity = zero2one_to_zero2inf(dist_from_body_ratios, method='arctanh', y_scale=lbs_precompute_settings["lbs_y_scale"]) # N_all_voxels
        weights_for_identity = 1 - torch.exp(-weights_for_identity)
        weights_for_identity = weights_for_identity.to(lbs_precompute_settings['precompute_device'])
        weights_for_identity = weights_for_identity * (1 - force_identity) + torch.ones_like(weights_for_identity) * force_identity
        # soft mixing
        weights_for_identity = weights_for_identity[None,:,None] # 1 x N_all_voxels x 1
        lbs_weights_per_voxel = identity_skin_weights * weights_for_identity + lbs_weights_per_voxel * (1 - weights_for_identity) # batch x N_all_voxels x J

    # method not implemented
    else:
        raise NotImplementedError

    ## reshape to 'grid' format & apply laplace smoothing
    # reshape skinning weights to grid format
    lbs_weights_per_voxel = lbs_weights_per_voxel / lbs_weights_per_voxel.sum(dim=-1, keepdim=True) # Normalize
    lbs_weights_per_voxel = lbs_weights_per_voxel.reshape(-1,N_x,N_y,N_z,N_joint) # NOTE batch x N_x x N_y x N_z x J
    # laplace smoothing for skinning weight
    if lbs_precompute_settings['lbs_smoothing_method'] == "laplace-smoothing":
        lbs_weights_per_voxel = laplace_smooth_weights(
            lbs_weights=lbs_weights_per_voxel, 
            smooth_times=lbs_precompute_settings['lbs_num_smoothing_times']
        )
    elif lbs_precompute_settings['lbs_smoothing_method'] == "laplace-smoothing+identity_thresh_preserve":
        lbs_weights_per_voxel = laplace_smooth_weights(
            lbs_weights=lbs_weights_per_voxel, 
            smooth_times=lbs_precompute_settings['lbs_num_smoothing_times'], 
            force_identity=force_identity
        )
    
    return lbs_weights_per_voxel.squeeze() # N_x x N_y x N_z x J



# [0,1] -> [0,inf] converter
def zero2one_to_zero2inf(ratio: torch.Tensor, method='tan', **kwargs):
	if method == 'tan':
		ratio = (ratio * np.pi / 2).clamp(min=0.0, max=np.pi / 2)
		output = kwargs['y_scale'] * torch.tan(ratio)
	elif method == 'arctanh':
		ratio = ratio.clamp(min=0.0, max=1.0)
		output = kwargs['y_scale'] * torch.arctanh(ratio)
	else:
		assert False, "Not implemented."
	return output



# laplace-smoothing skinning weights in grid
def laplace_smooth_weights(
        lbs_weights: torch.Tensor,
        smooth_times=3, 
        force_identity: torch.Tensor = None
    ):
    ## apply force identity for 'far' points
	if force_identity is not None:
		batch, N_x, N_y, N_z, N_joint = lbs_weights.shape
		force_identity = force_identity[None,:,None] # 1 x N_all_voxels x 1
		identity_weights = torch.zeros_like(lbs_weights).reshape(batch, N_x * N_y * N_z, N_joint); identity_weights[...,0] = 1 # batch x N_all_voxels x J
		
    ## apply laplace smoothing
	for _ in range(smooth_times):
		# laplace smoothing
		mean=(lbs_weights[:,2:,1:-1,1:-1,:]+lbs_weights[:,:-2,1:-1,1:-1,:]+\
			lbs_weights[:,1:-1,2:,1:-1,:]+lbs_weights[:,1:-1,:-2,1:-1,:]+\
			lbs_weights[:,1:-1,1:-1,2:,:]+lbs_weights[:,1:-1,1:-1,:-2,:])/6.0
		lbs_weights[:,1:-1,1:-1,1:-1,:]=(lbs_weights[:,1:-1,1:-1,1:-1,:]-mean)*0.7+mean
		
		# for outside identity_dist_threshold, update with identity 1
		if force_identity is not None:
			lbs_weights = lbs_weights.reshape(batch, N_x * N_y * N_z, N_joint) # batch x N_all_voxels x N_joint
			lbs_weights = lbs_weights * (1-force_identity) + identity_weights * force_identity  
			lbs_weights = lbs_weights.reshape(batch, N_x, N_y, N_z, N_joint)
		# normalize
		sums=lbs_weights.sum(dim=-1,keepdim=True)
		lbs_weights=lbs_weights/sums
                
	return lbs_weights