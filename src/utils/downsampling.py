"""
Utility functions for downsampling point clouds and flows to match encoded mask scale.
"""

import torch
import torch_scatter


def downsample_points_and_flows(point, pcs_list, flows_list):
    """
    Downsample point clouds and flows to match the encoded mask scale.
    
    Args:
        point: Point object from Sonata model output (contains pooling_inverse and pooling_parent)
        pcs_list: List of point clouds, each [N_i, 3]
        flows_list: List of flows, each [N_i, 3]
    
    Returns:
        downsampled_pcs_list: List of downsampled point clouds
        downsampled_flows_list: List of downsampled flows
        downsampling_info: Dict with information about downsampling
    """
    current_point = point
    downsampling_steps = []
    
    # Collect all pooling information (without popping to avoid modifying original)
    visited = set()
    while "pooling_parent" in current_point.keys() and "pooling_inverse" in current_point.keys():
        point_id = id(current_point)
        if point_id in visited:
            break
        visited.add(point_id)
        
        parent = current_point["pooling_parent"]
        inverse = current_point["pooling_inverse"]  # cluster: maps original -> downsampled
        downsampling_steps.append((parent, inverse))
        current_point = parent
    
    # Reverse the list to process from original to encoded scale
    downsampling_steps = list(reversed(downsampling_steps))
    
    # Also check for GridSampling inverse (from transform)
    grid_sampling_inverse = None
    if "inverse" in point.keys():
        grid_sampling_inverse = point["inverse"]
    
    # Start with original point clouds and flows
    current_pcs = pcs_list
    current_flows = flows_list
    
    # Apply downsampling steps
    for parent, inverse in downsampling_steps:
        # Downsample each point cloud and flow in the batch
        downsampled_pcs = []
        downsampled_flows = []
        
        start_idx = 0
        for pc, flow in zip(current_pcs, current_flows):
            n_points = pc.shape[0]
            end_idx = start_idx + n_points
            
            # Get the inverse indices for this batch item
            batch_inverse = inverse[start_idx:end_idx]
            
            # Find unique cluster IDs and their counts
            unique_clusters, cluster_counts = torch.unique(batch_inverse, return_counts=True)
            n_downsampled = len(unique_clusters)
            
            # Create index pointer for segment_csr
            idx_ptr = torch.cat([cluster_counts.new_zeros(1), torch.cumsum(cluster_counts, dim=0)])
            
            # Sort indices by cluster for segment_csr
            _, sorted_indices = torch.sort(batch_inverse)
            
            # Downsample coordinates using mean pooling
            downsampled_pc = torch_scatter.segment_csr(
                pc[sorted_indices], idx_ptr, reduce="mean"
            )
            
            # Downsample flows using mean pooling
            downsampled_flow = torch_scatter.segment_csr(
                flow[sorted_indices], idx_ptr, reduce="mean"
            )
            
            downsampled_pcs.append(downsampled_pc)
            downsampled_flows.append(downsampled_flow)
            
            start_idx = end_idx
        
        current_pcs = downsampled_pcs
        current_flows = downsampled_flows
    
    # Apply GridSampling inverse if present (this maps from pre-grid-sampling to post-grid-sampling)
    if grid_sampling_inverse is not None:
        # grid_sampling_inverse maps from original to grid-sampled
        # We need to apply this to the already downsampled point clouds
        final_pcs = []
        final_flows = []
        
        start_idx = 0
        for pc, flow in zip(current_pcs, current_flows):
            n_points = pc.shape[0]
            end_idx = start_idx + n_points
            
            # Get the inverse indices for this batch item
            batch_inverse = grid_sampling_inverse[start_idx:end_idx]
            
            # Find unique cluster IDs
            unique_clusters, cluster_counts = torch.unique(batch_inverse, return_counts=True)
            n_downsampled = len(unique_clusters)
            
            # Create index pointer
            idx_ptr = torch.cat([cluster_counts.new_zeros(1), torch.cumsum(cluster_counts, dim=0)])
            
            # Sort indices
            _, sorted_indices = torch.sort(batch_inverse)
            
            # Downsample using mean pooling
            downsampled_pc = torch_scatter.segment_csr(
                pc[sorted_indices], idx_ptr, reduce="mean"
            )
            downsampled_flow = torch_scatter.segment_csr(
                flow[sorted_indices], idx_ptr, reduce="mean"
            )
            
            final_pcs.append(downsampled_pc)
            final_flows.append(downsampled_flow)
            
            start_idx = end_idx
        
        current_pcs = final_pcs
        current_flows = final_flows
    
    return current_pcs, current_flows


def downsample_points_and_flows_simple(point, pcs_list, flows_list):
    """
    Simplified downsampling: directly use the final inverse if available.
    This is a faster version that assumes all downsampling can be done in one step.
    
    Args:
        point: Point object from Sonata model output
        pcs_list: List of point clouds, each [N_i, 3]
        flows_list: List of flows, each [N_i, 3]
    
    Returns:
        downsampled_pcs_list: List of downsampled point clouds
        downsampled_flows_list: List of downsampled flows
    """
    # Try to find the final inverse (from GridSampling or the deepest pooling)
    final_inverse = None
    
    # Check for GridSampling inverse first
    if "inverse" in point.keys():
        final_inverse = point["inverse"]
    else:
        # Find the deepest pooling_inverse
        current_point = point
        visited = set()
        while "pooling_parent" in current_point.keys() and "pooling_inverse" in current_point.keys():
            point_id = id(current_point)
            if point_id in visited:
                break
            visited.add(point_id)
            final_inverse = current_point["pooling_inverse"]
            current_point = current_point["pooling_parent"]
    
    if final_inverse is None:
        # No downsampling info, return original
        return pcs_list, flows_list
    
    # Apply downsampling
    downsampled_pcs = []
    downsampled_flows = []
    
    start_idx = 0
    for pc, flow in zip(pcs_list, flows_list):
        n_points = pc.shape[0]
        end_idx = start_idx + n_points
        
        # Get the inverse indices for this batch item
        batch_inverse = final_inverse[start_idx:end_idx]
        
        # Find unique cluster IDs
        unique_clusters, cluster_counts = torch.unique(batch_inverse, return_counts=True)
        n_downsampled = len(unique_clusters)
        
        # Create index pointer
        idx_ptr = torch.cat([cluster_counts.new_zeros(1), torch.cumsum(cluster_counts, dim=0)])
        
        # Sort indices
        _, sorted_indices = torch.sort(batch_inverse)
        
        # Downsample using mean pooling
        downsampled_pc = torch_scatter.segment_csr(
            pc[sorted_indices], idx_ptr, reduce="mean"
        )
        downsampled_flow = torch_scatter.segment_csr(
            flow[sorted_indices], idx_ptr, reduce="mean"
        )
        
        downsampled_pcs.append(downsampled_pc)
        downsampled_flows.append(downsampled_flow)
        
        start_idx = end_idx
    
    return downsampled_pcs, downsampled_flows


