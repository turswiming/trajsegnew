"""
Input/Output utility functions.
"""

import os
import numpy as np
import torch


def save_segmentation_results(pcs_list, masks_list, flows_list, output_dir, step, batch_idx=0):
    """
    Save point cloud and segmentation results for visualization.
    
    Args:
        pcs_list: List of point clouds [N, 3]
        masks_list: List of mask logits [K, N]
        flows_list: List of flows [N, 3]
        output_dir: Directory to save results
        step: Current training step
        batch_idx: Which sample in batch to save (default: 0)
    
    Returns:
        Path to saved directory, or None if failed
    """
    save_dir = os.path.join(output_dir, f"step_{step:06d}")
    os.makedirs(save_dir, exist_ok=True)
    
    if batch_idx < len(pcs_list):
        pc = pcs_list[batch_idx].detach().cpu().numpy()
        mask = masks_list[batch_idx].detach().cpu().numpy()
        flow = flows_list[batch_idx].detach().cpu().numpy()
        
        seg_labels = np.argmax(mask, axis=0)
        
        np.savez(
            os.path.join(save_dir, "sample.npz"),
            points=pc,
            mask_logits=mask,
            seg_labels=seg_labels,
            flow=flow
        )
        
        return save_dir
    return None
