#!/usr/bin/env python3
"""
Script to analyze the average number of instances per frame in the eval mask
for the AV2 segmentation validation dataset.
"""

import sys
import os
sys.path.append('/root/autodl-tmp/trajsegnew/src')

import torch
from datasets.seg_dataset import SegValDataset
import numpy as np

def main():
    data_dir = "/root/autodl-tmp/av2_seg/seg_val"
    
    # Create dataset
    dataset = SegValDataset(data_dir)
    
    print(f"Dataset size: {len(dataset)}")
    
    instance_counts = []
    
    # For quick analysis, limit to first 100 samples
    num_samples = min(100, len(dataset))
    print(f"Analyzing first {num_samples} samples...")
    
    for i in range(num_samples):
        sample = dataset[i]
        
        instance_label = sample['instance_label'].numpy()
        eval_mask = sample['eval_mask'].numpy()
        
        # Debug shapes
        if i == 0:
            print(f"instance_label shape: {instance_label.shape}")
            print(f"eval_mask shape: {eval_mask.shape}")
            print(f"eval_mask dtype: {eval_mask.dtype}")
        
        # Only consider points in eval_mask
        eval_mask = eval_mask.squeeze()  # Ensure it's 1D
        valid_instances = instance_label[eval_mask]
        
        # Get unique instance IDs (ignore -1 or invalid)
        unique_instances = np.unique(valid_instances)
        # Filter out invalid instances (assuming -1 is invalid)
        valid_unique_instances = unique_instances[unique_instances >= 0]
        
        num_instances = len(valid_unique_instances)
        instance_counts.append(num_instances)
        
        if i % 100 == 0:
            print(f"Processed {i}/{num_samples} samples")
    
    # Calculate statistics
    instance_counts = np.array(instance_counts)
    mean_instances = np.mean(instance_counts)
    median_instances = np.median(instance_counts)
    min_instances = np.min(instance_counts)
    max_instances = np.max(instance_counts)
    
    print("\nStatistics:")
    print(f"Average instances per frame: {mean_instances:.2f}")
    print(f"Median instances per frame: {median_instances}")
    print(f"Min instances per frame: {min_instances}")
    print(f"Max instances per frame: {max_instances}")
    print(f"Total samples: {len(instance_counts)}")

if __name__ == "__main__":
    main()