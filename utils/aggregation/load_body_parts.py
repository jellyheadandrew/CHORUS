import json
from easydict import EasyDict

import numpy as np
import torch

from constants.aggr_settings.semantic_clustering import EPSILON_DEFAULT, BODY_PART_CORRESPONDENCE



def load_body_parts(aggr_core, verbose=False):
    # print status
    if verbose:
        print("Creating interaction regions for clustering!!!")
    
    ## iterate for all tags in 'aggr_core'
    for tag in aggr_core.keys():
        # no need to set body-parts for 'GLOBAL'
        if tag == 'GLOBAL':
            continue
        
        # prepare data
        v_template_base = aggr_core[tag].v_template_holistic
        body_segmentation_json_pth = aggr_core[tag].body_segmentation_json_pth
        body_part_define_method = aggr_core[tag].body_part_define_method

        # get body-part mesh-vertex indices for each tag
        if tag == "SMPL":
            body_parts, epsilon_per_body_part, aggr_world_name_per_body_parts = load_segmentation_smpl(
                body_segmentation_json_pth = body_segmentation_json_pth,
                body_part_define_method = body_part_define_method,
                v_template=v_template_base,
                verbose=verbose,
            )
            
        # update the 'aggr_core' body_parts, epsilon_per_body_part, aggr_world_name_per_body_parts
        aggr_core[tag].body_parts = body_parts
        aggr_core[tag].epsilon_per_body_part = epsilon_per_body_part # corresponds to epsilon (interaction-dist-threshold) per body-part, per tag
        aggr_core[tag].aggr_world_name_per_body_parts = aggr_world_name_per_body_parts # tags per body-parts | used for which tag to compute interaction-region

        # check if duplicate body_part names exist
        body_part_names = sorted(list(aggr_core[tag].body_parts.keys()))
        assert sorted(list(set(body_part_names))) == body_part_names, f"No duplicate body-part name allowed. Check the aggregation settings"
        aggr_core[tag].body_part_names = body_part_names # originally 'all_cluster_keys'

        # print body-part-names
        if verbose:
            print(f"Body part names: {body_part_names}")

        ## iterate through body parts
        interaction_regions = EasyDict()
        for body_part_name in body_parts.keys():
            # generate interaction region
            assert body_part_name not in interaction_regions, "Duplicate part_name in 'inter_regions'"
            interaction_regions[body_part_name] = EasyDict()

            ## iterate through all tags
            for _tag in epsilon_per_body_part[body_part_name].keys():
                # first, translate the '_tag' canon-grid to 'tag' canon-grid
                aligned_canon_grid = \
                    aggr_core[_tag].canon_grid + \
                    aggr_core[_tag].canoncam2smpl_trans[:,None,None,None] - \
                    aggr_core[tag].canoncam2smpl_trans[:,None,None,None]

                # second, compute the interaction-region based on inter_dist_thresh ('epsilon')
                interaction_region_voxel_indices = compute_interaction_region(
                    canon_grid = aligned_canon_grid,
                    v_template = aggr_core[tag].v_template_holistic,
                    body_part_vert_indices = body_parts[body_part_name],
                    inter_dist_thresh = epsilon_per_body_part[body_part_name][tag],
                    verbose=verbose,
                    **aggr_core.GLOBAL.interaction_region_precompute_settings
                )

                # third, update the 'interaction_regions' with voxel-indices
                interaction_regions[body_part_name][tag] = interaction_region_voxel_indices

        ## update 'aggr_core' with 'interaction_regions'
        aggr_core[tag].interaction_regions = interaction_regions        
        
    # print ending status of loading body parts
    if verbose:
        print("Loading Body Parts Finished!!!")
    
    return aggr_core        



def compute_interaction_region(
    canon_grid, # 3xNxNxN
    v_template, # Vx3
    body_part_vert_indices, # List of indices
    inter_dist_thresh, # Float value
    precompute_device="cpu",
    chunk_size=50000,
    verbose=False,
    **kwargs
    ):
    # dict to save 'interaction_regions' to pass or block (each values are list of 3D voxel indices, close to mesh)
    interaction_regions = {}
    
    # 'pass' or 'block' images if 3D rays pass through interaction_region
    for pass_or_block in ["pass", "block"]:         
        ## iterate for 'pass' and 'block' 
        interaction_region = []
        for _body_part_vert_indices in body_part_vert_indices[pass_or_block]:
            # shape of canonical voxel-grid: (3, N_x, N_y, N_z)
            N_x, N_y, N_z = canon_grid.shape[1:]

            # N_body_part_vertices x N_all_voxels x 3
            displacements = \
                canon_grid.to(precompute_device).reshape(3,-1).T[None,:,:] \
                - v_template.to(precompute_device)[_body_part_vert_indices][:,None,:]
            
            ## precomputation kills cpu if implemented naively: 'precompute per chunks'
            # placeholder to save distance between 'canon-voxel' and 'body_part vertex'
            distances = torch.empty(displacements.shape[:2], dtype=torch.float32) # N_body_part_vertices x N_all_voxels
            
            # iterate for all 'chunks'
            displacements_done = 0
            while displacements_done < displacements.shape[1]:
                # print progress
                if verbose:
                    print("\n===== precomputing interaction regions ... =====")
                    print("[interaction-region] precomputing: ", displacements_done)
                
                # which chunk to precompute
                start_idx = displacements_done
                end_idx = min(displacements_done + chunk_size, displacements.shape[1])

                # compute 'distance between 'body-part vertices' and 'canon-grid voxels'
                distances[:,start_idx:end_idx] = displacements[:,start_idx:end_idx].square().sum(dim=-1).sqrt()

                # update 'displacements_done' to move to next chunk
                displacements_done = end_idx
                
            # remove used 'displacements' variable & print 'end' for precomputation
            del displacements
            if verbose:
                print("===== precomputing interaction regions: done! =====\n")
            

            ## reshape
            distances = distances.reshape(distances.shape[0],N_x,N_y,N_z) # shape: N_body_part_vertices, N_x, N_y, N_z

            # boolean array that checks whether voxel is inside 'interaction_region' or not
            interaction_region_grid = torch.where(distances.min(dim=0)[0] < inter_dist_thresh, 1, 0).type(torch.float32) # shape: N_x x N_y x N_z
            
            # N_inter_points x 3, long
            interaction_region_indices = torch.nonzero(interaction_region_grid)

            # update interaction_region
            interaction_region.append(interaction_region_indices) 

        # update interaction_regions
        interaction_regions[pass_or_block] = interaction_region

    return interaction_regions # dict{list[N_inter_points x 3]} / dtype: long



def load_segmentation_smpl(
    body_segmentation_json_pth,
    body_part_define_method,
    verbose=False,
    **kwargs
    ):
    # get body_parts indices
    with open(body_segmentation_json_pth, "r") as rf:
        body_segmentation_loaded = json.load(rf)

    # we must return these
    body_parts = dict()
    epsilon_per_body_part = dict()
    aggr_world_name_per_body_parts = dict()

    # print body-part-define method
    if verbose:
        print("body_part_define_method: ", body_part_define_method)

    ## merge 'body_segmentation's to obtain 'body_part' mesh vertex indices 
    for body_part_name, data in BODY_PART_CORRESPONDENCE[body_part_define_method].items():
        # unpack data
        correspondence_map = data['correspondence_map']
        interaction_threshold_per_world = data['interaction_threshold_per_world']

        ## there are two types of body parts
        # 'pass' -> at least one vertex must lie within 2D occupancy
        # 'block' -> any vertex must not lie within 2D occupancy
        body_parts[body_part_name] = {'pass': [], 'block': []}
        
        # merge to get body part (pass)
        for correspondence in correspondence_map['pass']:
            # this is a list of list. Each list contains vertex indices (empty list means no body-part clustering)
            body_parts[body_part_name]['pass'].append(list())
            # for distinct list of indices, each list must contain at least one point that lies within 2D occupancy
            for body_seg_key in correspondence:
                body_parts[body_part_name]['pass'][-1] += body_segmentation_loaded[body_seg_key]
            body_parts[body_part_name]['pass'][-1] = list(set(body_parts[body_part_name]['pass'][-1]))

        # merge to get body part (block)
        for correspondence in correspondence_map['block']:
            # this is a list of list. Each list contains vertex indices (empty list means no body-part clustering)
            body_parts[body_part_name]['block'].append(list())
            # for distinct list of indices, each list must contain all points that does not lie within 2D occupancy
            for body_seg_key in correspondence:
                body_parts[body_part_name]['block'][-1] += body_segmentation_loaded[body_seg_key]
            body_parts[body_part_name]['block'][-1] = list(set(body_parts[body_part_name]['block'][-1]))
            
        # save epsilon (interaction-distance threshold) per body-part
        epsilon_per_body_part[body_part_name] = interaction_threshold_per_world

        # world-tag per body parts
        aggr_world_name_per_body_parts[body_part_name] = list(interaction_threshold_per_world.keys())

    return fill_in_missing_parts(body_parts, epsilon_per_body_part, aggr_world_name_per_body_parts)



def fill_in_missing_parts(body_parts, epsilon_per_body_part, aggr_world_name_per_body_parts):
    # Check for invalid keys in 'epsilon_per_body_part'
    for k in epsilon_per_body_part.keys():
        if k not in body_parts.keys():
            assert False, "Invalid key in 'epsilon_per_body_part'"

    # Check for invalid keys in 'aggr_world_name_per_body_parts
    for k in aggr_world_name_per_body_parts.keys():
        if k not in body_parts.keys():
            assert False, "Invalid key in 'aggr_world_name_per_body_parts'"

    # If keys for 'epsilon_per_body_part' is differnt from 'body_parts', fill default values
    for k in body_parts.keys():
        if k not in epsilon_per_body_part.keys():
            epsilon_per_body_part[k] = {'SMPL': EPSILON_DEFAULT}

    # If keys for 'aggr_world_name_per_body_parts' is different from 'body_parts', fill with default values
    for k in body_parts.keys():
        if k not in aggr_world_name_per_body_parts.keys():
            aggr_world_name_per_body_parts[k] = ['SMPL']
        else:
            # If tag is not added to aggr_world_name_per_body_parts, add it
            if 'SMPL' not in aggr_world_name_per_body_parts[k]:
                aggr_world_name_per_body_parts[k].append('SMPL')

    # Wrap the lists with integer values as a list for changed format of inter-region computation
    for k in body_parts.keys():
        if type(body_parts[k]) == list:
            assert len(body_parts[k]) >= 1
            if type(body_parts[k][0]) == int:
                body_parts[k] = [body_parts[k]]
            else:
                for seg_part in body_parts[k]:
                    assert type(seg_part) == list
        elif type(body_parts[k]) == dict:
            assert "pass" in body_parts[k].keys()
            assert "block" in body_parts[k].keys()
            assert len(body_parts[k]["pass"]) >= 1
            # body_parts[k] = 

    # If not specified, then Wrap the "list of lists" to a dictionary of "pass" and "block"
    for k in body_parts.keys():
        if type(body_parts[k]) == dict:
            assert "pass" in body_parts[k]
            assert "block" in body_parts[k]
            assert type(body_parts[k]["pass"]) == list 
            if len(body_parts[k]["pass"]) > 0: assert type(body_parts[k]["pass"][0]) == list
            assert type(body_parts[k]["block"]) == list
            if len(body_parts[k]["block"]) > 0: assert type(body_parts[k]["block"][0]) == list
        elif type(body_parts[k]) == list and type(body_parts[k][0]) == list:
            body_parts[k] = {"pass": body_parts[k], "block": []}
        else:
            assert False, f"body_parts here looks like this for key: {k} -> {body_parts[k]}"

    # final check for invalid
    assert tuple(sorted(list(body_parts.keys()))) == tuple(sorted(list(epsilon_per_body_part.keys())))
    assert tuple(sorted(list(body_parts.keys()))) == tuple(sorted(list(aggr_world_name_per_body_parts.keys())))
    for k in body_parts.keys():
        assert tuple(sorted(list(aggr_world_name_per_body_parts[k]))) == tuple(sorted(list(epsilon_per_body_part[k].keys()))), "The tags under 'epsilon_per_body_part' and 'aggr_world_name_per_body_parts' must be same."

    return body_parts, epsilon_per_body_part, aggr_world_name_per_body_parts