"""
Utility functions for upsampling masks from encoded scale to original point cloud scale.
Based on Sonata's upsampling approach from README.
"""

import torch


def upsample_masks_from_point(point, masks, original_num_points):
    """
    Upsample masks from encoded scale back to original point cloud scale.
    Follows the approach described in Sonata README.
    
    Args:
        point: Point object from Sonata model output (contains pooling_parent and pooling_inverse)
        masks: [K, N_encoded] segmentation masks at encoded scale
        original_num_points: Number of points in original point cloud
    
    Returns:
        [K, N_original] upsampled masks
    """
    current_point = point
    current_masks = masks  # [K, N_encoded]
    
    # First 2 levels: use concatenation approach (but for masks we just upsample)
    for _ in range(2):
        if "pooling_parent" not in current_point.keys() or "pooling_inverse" not in current_point.keys():
            break
        
        parent = current_point.pop("pooling_parent")
        inverse = current_point.pop("pooling_inverse")
        
        # Upsample masks: use inverse to map from child to parent
        N_parent = parent.feat.shape[0] if hasattr(parent, 'feat') else parent.coord.shape[0]
        
        # Map child masks to parent using inverse
        # Use indexing directly to preserve gradients
        if inverse is not None and inverse.shape[0] == current_masks.shape[1]:
            # inverse[i] tells which child point corresponds to parent point i
            # We need to map: for each parent point, use the mask from its child point
            if inverse.max() < current_masks.shape[1]:
                # Direct indexing preserves gradients
                current_masks = current_masks[:, inverse]
            else:
                # Handle out-of-bounds: use scatter or direct indexing where possible
                # For gradient preservation, we'll use indexing with valid mask
                valid = inverse < current_masks.shape[1]
                # Create result tensor, but use indexing to preserve gradient
                # We need to map valid indices
                if valid.sum() == N_parent:
                    # All indices valid, use direct indexing
                    current_masks = current_masks[:, inverse]
                else:
                    # Some indices invalid - clip to valid range to preserve gradient
                    inverse_clipped = torch.clamp(inverse, 0, current_masks.shape[1] - 1)
                    current_masks = current_masks[:, inverse_clipped]
        else:
            # If inverse doesn't match, pad with zeros (this will break gradient, but it's a fallback)
            if current_masks.shape[1] < N_parent:
                pad_size = N_parent - current_masks.shape[1]
                current_masks = torch.cat([
                    current_masks,
                    torch.zeros(current_masks.shape[0], pad_size,
                              device=current_masks.device,
                              dtype=current_masks.dtype,
                              requires_grad=current_masks.requires_grad)
                ], dim=1)
        current_point = parent
    
    # Remaining levels: simple indexing
    while "pooling_parent" in current_point.keys() and "pooling_inverse" in current_point.keys():
        parent = current_point.pop("pooling_parent")
        inverse = current_point.pop("pooling_inverse")
        
        N_parent = parent.feat.shape[0] if hasattr(parent, 'feat') else parent.coord.shape[0]
        
        if inverse is not None:
            if inverse.shape[0] == current_masks.shape[1]:
                if inverse.max() < current_masks.shape[1]:
                    # Direct indexing preserves gradients
                    current_masks = current_masks[:, inverse]
                else:
                    # Handle out-of-bounds: use indexing with valid mask
                    # Try to use direct indexing if possible to preserve gradient
                    valid = inverse < current_masks.shape[1]
                    if valid.all():
                        # All valid, use direct indexing
                        current_masks = current_masks[:, inverse]
                    else:
                        # Some invalid - clip to valid range to preserve gradient
                        inverse_clipped = torch.clamp(inverse, 0, current_masks.shape[1] - 1)
                        current_masks = current_masks[:, inverse_clipped]
            else:
                # Fallback: pad or truncate
                if current_masks.shape[1] < N_parent:
                    pad_size = N_parent - current_masks.shape[1]
                    current_masks = torch.cat([
                        current_masks,
                        torch.zeros(current_masks.shape[0], pad_size,
                                  device=current_masks.device,
                                  dtype=current_masks.dtype,
                                  requires_grad=current_masks.requires_grad)
                    ], dim=1)
                else:
                    current_masks = current_masks[:, :N_parent]
        current_point = parent
    
    # Final step: map back to original point cloud using inverse (from GridSampling)
    if "inverse" in current_point.keys():
        inverse = current_point["inverse"]
        N_original = min(original_num_points, inverse.shape[0])
        
        # inverse[i] tells which encoded point corresponds to original point i
        # So we use: masks_original[i] = masks_encoded[inverse[i]]
        # This is exactly: masks_original = masks_encoded[inverse]
        if inverse.max() < current_masks.shape[1]:
            # Safe indexing
            final_masks = current_masks[:, inverse[:N_original]]
        else:
            # Handle out-of-bounds: clip indices to preserve gradient
            inverse_clipped = torch.clamp(inverse[:N_original], 0, current_masks.shape[1] - 1)
            final_masks = current_masks[:, inverse_clipped]
            # If we had to clip, we might have fewer points - pad if needed
            if final_masks.shape[1] < N_original:
                pad_size = N_original - final_masks.shape[1]
                final_masks = torch.cat([
                    final_masks,
                    torch.zeros(final_masks.shape[0], pad_size,
                              device=final_masks.device,
                              dtype=final_masks.dtype,
                              requires_grad=False)  # Padding doesn't need gradient
                ], dim=1)
        
        return final_masks
    
    # If no inverse found, ensure dimensions match
    if current_masks.shape[1] != original_num_points:
        if current_masks.shape[1] < original_num_points:
            # Pad (preserve gradient by setting requires_grad)
            final_masks = torch.zeros(
                current_masks.shape[0], original_num_points,
                device=current_masks.device,
                dtype=current_masks.dtype,
                requires_grad=current_masks.requires_grad
            )
            final_masks[:, :current_masks.shape[1]] = current_masks
            return final_masks
        else:
            # Truncate (this preserves gradient)
            return current_masks[:, :original_num_points]
    
    return current_masks


def upsample_masks_simple(masks, upsampling_info, original_num_points):
    """
    Simplified upsampling using upsampling_info list.
    This is a fallback if we can't access the Point object directly.
    
    Args:
        masks: [K, N_encoded] segmentation masks at encoded scale
        upsampling_info: List of (parent, inverse) tuples from model output
        original_num_points: Number of points in original point cloud
    
    Returns:
        [K, N_original] upsampled masks
    """
    if not upsampling_info:
        # No upsampling info, try to match dimensions
        if masks.shape[1] != original_num_points:
            if masks.shape[1] < original_num_points:
                # Pad (preserve gradient)
                upsampled = torch.zeros(
                    masks.shape[0], original_num_points,
                    device=masks.device,
                    dtype=masks.dtype,
                    requires_grad=masks.requires_grad
                )
                upsampled[:, :masks.shape[1]] = masks
                return upsampled
            else:
                return masks[:, :original_num_points]
        return masks
    
    # Try to find GridSampling inverse (last one with parent=None)
    final_inverse = None
    for parent, inverse in reversed(upsampling_info):
        if parent is None and inverse is not None:
            final_inverse = inverse
            break
    
    if final_inverse is not None:
        # Use inverse to map from encoded back to original
        # inverse[i] tells which encoded point corresponds to original point i
        N_original = min(original_num_points, final_inverse.shape[0])
        
        if final_inverse.max() < masks.shape[1]:
            # Safe indexing: masks_original = masks_encoded[inverse]
            upsampled_masks = masks[:, final_inverse[:N_original]]
        else:
            # Handle out-of-bounds (preserve gradient)
            upsampled_masks = torch.zeros(
                masks.shape[0], N_original,
                device=masks.device,
                dtype=masks.dtype,
                requires_grad=masks.requires_grad
            )
            valid = (final_inverse[:N_original] < masks.shape[1])
            upsampled_masks[:, valid] = masks[:, final_inverse[:N_original][valid]]
        
        return upsampled_masks
    
    # Fallback: pad or truncate
    if masks.shape[1] < original_num_points:
        upsampled = torch.zeros(
            masks.shape[0], original_num_points,
            device=masks.device,
            dtype=masks.dtype,
            requires_grad=masks.requires_grad
        )
        upsampled[:, :masks.shape[1]] = masks
        return upsampled
    else:
        return masks[:, :original_num_points]

