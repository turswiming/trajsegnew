import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import glob
import os
import logging
import pickle

Split_into_4_part = True

class SegTrainDataset(Dataset):
    """
    Training dataset for Trajectory prediction / Segmentation.
    
    Expected Dataset Structure:
    - data_dir should contain .h5 files matching "*_trajectories.h5".
    
    H5 File Structure (Original Format):
    Each .h5 file contains groups named by keys (likely timestamps).
    Group structure:
        - 'SSL_trajectory_step_0': [N, 3] Point cloud coordinates at step 0 (float).
        - 'SSL_trajectory_step_1': [N, 3] Point cloud coordinates at step 1 (float). Used to compute flow.
        - 'SSL_long_term_trajectory': [N, T, 3] or similar. Future trajectory of points.
    
    Returns:
        Dictionary with keys:
        - 'pc': [N, 3] torch.FloatTensor
        - 'flow': [N, 3] torch.FloatTensor (calculated as step_1 - step_0)
        - 'trajectories': [N, ...] torch.FloatTensor
    """
    def __init__(self, data_dir, num_points=32768, transform=None):
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.files = glob.glob(os.path.join(data_dir, "*_trajectories.h5"))
        self.items = []
        
        print(f"Found {len(self.files)} files in {data_dir}")
        for f in self.files:
            with h5py.File(f, 'r') as hf:
                keys = list(hf.keys())
                for k in keys:
                    self.items.append((f, k))
        print(f"Found {len(self.items)} samples in {data_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_path, group_name = self.items[idx]
        
        with h5py.File(file_path, 'r') as hf:
            group = hf[group_name]
            
            # 1. Point Cloud: Use step 0
            points = group['SSL_trajectory_step_0'][:]
            
            # 2. Scene Flow
            # Ideally use 'SSL_sceneflow_0', but due to shape mismatch (likely valid mask missing),
            # we compute flow from trajectory step 1 - step 0 which is consistent with points.
            # Using SSL label (trajectory step 1) to derive flow.
            if 'SSL_trajectory_step_1' in group:
                p1 = group['SSL_trajectory_step_1'][:]
                flow = p1 - points
            else:
                # Fallback
                flow = points  # Should not happen
            # 3. Trajectory
            if 'SSL_long_term_trajectory' in group:
                trajectory = group['SSL_long_term_trajectory'][:]
            else:
                 # Fallback
                 trajectory = points # Should not happen
            if Split_into_4_part:
                # Split into 4 parts along x,y axis
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                #求质心
                centroid = np.mean(points, axis=0)
                #以质心分成4份
                part1_mask = (x_coords < centroid[0]) & (y_coords < centroid[1])
                part2_mask = (x_coords >= centroid[0]) & (y_coords < centroid[1])
                part3_mask = (x_coords < centroid[0]) & (y_coords >= centroid[1])
                part4_mask = (x_coords >= centroid[0]) & (y_coords >= centroid[1])
                #random seletct one part
                selected_part = idx % 4
                if selected_part == 0:
                    points = points[part1_mask]
                    flow = flow[part1_mask]
                    trajectory = trajectory[part1_mask]
                elif selected_part == 1:
                    points = points[part2_mask]
                    flow = flow[part2_mask]
                    trajectory = trajectory[part2_mask]
                elif selected_part == 2:
                    points = points[part3_mask]
                    flow = flow[part3_mask]
                    trajectory = trajectory[part3_mask]
                elif selected_part == 3:
                    points = points[part4_mask]
                    flow = flow[part4_mask]
                    trajectory = trajectory[part4_mask]
        obj = {
            'pc': torch.from_numpy(points).float(),
            'flow': torch.from_numpy(flow).float(),
            'trajectories': torch.from_numpy(trajectory).float()
        }
        if Split_into_4_part:
            obj['part'] = selected_part
        return  obj

class SegValDataset(Dataset):
    """
    Validation dataset for Segmentation (AV2).
    
    Expected Dataset Structure:
    - data_dir should contain .h5 files and an 'index_eval.pkl' file.
    - 'index_eval.pkl': A list of [filename_stem, timestamp_key] pairs identifying valid evaluation frames.
    
    H5 File Structure:
    Each .h5 file contains groups named by timestamp.
    Group structure:
        - 'lidar': [N, 3] Point cloud coordinates (float).
        - 'flow': [N, 3] Scene flow vectors (float).
        - 'instance_label': [N] Instance IDs for segmentation (long).
        - 'eval_mask': [N] (Optional) Boolean mask indicating points to be evaluated.
    
    Returns:
        Dictionary with keys:
        - 'pc': [N, 3] torch.FloatTensor
        - 'flow': [N, 3] torch.FloatTensor
        - 'instance_label': [N] torch.LongTensor
        - 'eval_mask': [N] torch.BoolTensor
    """
    def __init__(self, data_dir, num_points=32768, transform=None):
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.items = []
        
        index_file = os.path.join(data_dir, "index_eval.pkl")
        if os.path.exists(index_file):
            print(f"Loading validation index from {index_file}...")
            with open(index_file, 'rb') as f:
                index_list = pickle.load(f)
            
            for stem, key in index_list:
                file_path = os.path.join(data_dir, f"{stem}.h5")
                self.items.append((file_path, key))
            print(f"Loaded {len(self.items)} samples from index file.")
        else:
            print(f"Index file {index_file} not found, scanning directory instead...")
            files = glob.glob(os.path.join(data_dir, "*.h5"))
            for f in files:
                try:
                    with h5py.File(f, 'r') as hf:
                        keys = list(hf.keys())
                        for k in keys:
                            self.items.append((f, k))
                except Exception as e:
                    print(f"Error reading {f}: {e}")
            print(f"Found {len(self.items)} validation samples in {data_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_path, group_name = self.items[idx]
        
        with h5py.File(file_path, 'r') as hf:
            group = hf[group_name]
            
            points = group['lidar'][:]
            flow = group['flow'][:]
            instance_label = group['instance_label'][:]
            
            if 'eval_mask' in group:
                eval_mask = group['eval_mask'][:]
            else:
                eval_mask = np.ones(points.shape[0], dtype=bool)
        if Split_into_4_part:
            # Split into 4 parts along x,y axis
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            #求质心
            centroid = np.mean(points, axis=0)
            #以质心分成4份
            part1_mask = (x_coords < centroid[0]) & (y_coords < centroid[1])
            part2_mask = (x_coords >= centroid[0]) & (y_coords < centroid[1])
            part3_mask = (x_coords < centroid[0]) & (y_coords >= centroid[1])
            part4_mask = (x_coords >= centroid[0]) & (y_coords >= centroid[1])
            #random seletct one part
            selected_part = idx % 4
            if selected_part == 0:
                points = points[part1_mask]
                flow = flow[part1_mask]
                instance_label = instance_label[part1_mask]
                eval_mask = eval_mask[part1_mask]
            elif selected_part == 1:
                points = points[part2_mask]
                flow = flow[part2_mask]
                instance_label = instance_label[part2_mask]
                eval_mask = eval_mask[part2_mask]
            elif selected_part == 2:
                points = points[part3_mask]
                flow = flow[part3_mask]
                instance_label = instance_label[part3_mask]
                eval_mask = eval_mask[part3_mask]
            elif selected_part == 3:
                points = points[part4_mask]
                flow = flow[part4_mask]
                instance_label = instance_label[part4_mask]
                eval_mask = eval_mask[part4_mask]
        obj = {
            'pc': torch.from_numpy(points).float(),
            'flow': torch.from_numpy(flow).float(),
            'instance_label': torch.from_numpy(instance_label).long(),
            'eval_mask': torch.from_numpy(eval_mask).bool()
        }
        if Split_into_4_part:
            obj['part'] = selected_part
        return obj

class DebugTrainDataset(SegTrainDataset):
    def __init__(self, data_dir, num_points=32768):
        super().__init__(data_dir, num_points)
        if len(self.items) > 100:
            self.items = self.items[:100]

class DebugValDataset(SegValDataset):
    def __init__(self, data_dir, num_points=32768):
        super().__init__(data_dir, num_points)
        if len(self.items) > 10:
            self.items = self.items[:10]
