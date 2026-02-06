"""
Data utility functions for point cloud processing.
"""

import torch


def collate_fn(batch):
    """
    Custom collate function for variable-length point clouds.
    Returns lists instead of stacking tensors.
    
    Args:
        batch: List of samples, each containing 'pc', 'flow', and optionally 
               'instance_label' and 'trajectories'
    
    Returns:
        Dictionary with batched data as lists
    """
    result = {
        'pc': [item['pc'] for item in batch],
        'flow': [item['flow'] for item in batch]
    }
    
    # Include instance_label if present in any sample
    if 'instance_label' in batch[0]:
        result['instance_label'] = [item['instance_label'] for item in batch]
    
    # Include trajectories if present in any sample
    if 'trajectories' in batch[0]:
        result['trajectories'] = [item['trajectories'] for item in batch]
    
    return result


def prepare_point_dict(pcs_list, device):
    """
    Prepare point dictionary for model input.
    
    Args:
        pcs_list: List of point clouds, each [N_i, 3], or a single tensor [N, 3]
        device: Device to move data to
    
    Returns:
        Dictionary with point data, with correct batch indices for each point cloud
    """
    if isinstance(pcs_list, list):
        all_pcs = torch.cat(pcs_list, dim=0).to(device)
        batch_indices = []
        for batch_idx, pc in enumerate(pcs_list):
            n_points = pc.shape[0] if isinstance(pc, torch.Tensor) else len(pc)
            batch_indices.append(torch.full((n_points,), batch_idx, dtype=torch.long, device=device))
        batch = torch.cat(batch_indices, dim=0)
    else:
        all_pcs = pcs_list.to(device)
        batch = torch.zeros(all_pcs.shape[0], dtype=torch.long, device=device)
    
    coords = all_pcs
    feats = all_pcs  # Use coordinates as features
    
    return {
        'coord': coords,
        'feat': feats,
        'batch': batch
    }
