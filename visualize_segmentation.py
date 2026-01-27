"""
Visualization script for point cloud segmentation results.
Usage:
    python visualize_segmentation.py --input outputs/20250613_220530/step_000100/sample.npz
    python visualize_segmentation.py --input_dir outputs/20250613_220530 --step 100
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


def get_distinct_colors(n):
    """Generate n distinct colors for segmentation visualization."""
    if n <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n]
    elif n <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n]
    else:
        # Use hsv colormap for more colors
        colors = plt.cm.hsv(np.linspace(0, 0.9, n))
    return colors


def visualize_segmentation(points, seg_labels, flow=None, title="Point Cloud Segmentation", 
                           save_path=None, show=True, elev=30, azim=45):
    """
    Visualize point cloud with segmentation labels.
    
    Args:
        points: [N, 3] point coordinates
        seg_labels: [N] segmentation labels
        flow: [N, 3] optional flow vectors
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
    """
    # Get unique labels and colors
    unique_labels = np.unique(seg_labels)
    n_labels = len(unique_labels)
    colors = get_distinct_colors(n_labels)
    
    # Create color array for each point
    point_colors = np.zeros((len(points), 4))
    for i, label in enumerate(unique_labels):
        mask = seg_labels == label
        point_colors[mask] = colors[i]
    
    # Create figure
    if flow is not None:
        fig = plt.figure(figsize=(16, 6))
        
        # Segmentation view
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=point_colors, s=1, alpha=0.8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Segmentation ({n_labels} segments)')
        ax1.view_init(elev=elev, azim=azim)
        
        # Flow magnitude view
        ax2 = fig.add_subplot(132, projection='3d')
        flow_mag = np.linalg.norm(flow, axis=1)
        scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                             c=flow_mag, cmap='viridis', s=1, alpha=0.8)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Flow Magnitude')
        ax2.view_init(elev=elev, azim=azim)
        plt.colorbar(scatter, ax=ax2, shrink=0.5, label='Flow Magnitude')
        
        # Top-down view with flow arrows (subsample for clarity)
        ax3 = fig.add_subplot(133)
        subsample = max(1, len(points) // 1000)
        ax3.scatter(points[::subsample, 0], points[::subsample, 1], 
                   c=point_colors[::subsample], s=2, alpha=0.6)
        # Draw flow arrows
        arrow_subsample = max(1, len(points) // 200)
        ax3.quiver(points[::arrow_subsample, 0], points[::arrow_subsample, 1],
                  flow[::arrow_subsample, 0], flow[::arrow_subsample, 1],
                  color='red', alpha=0.5, scale=50, width=0.002)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Top-down View with Flow')
        ax3.set_aspect('equal')
    else:
        fig = plt.figure(figsize=(12, 5))
        
        # 3D view
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=point_colors, s=1, alpha=0.8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Segmentation ({n_labels} segments)')
        ax1.view_init(elev=elev, azim=azim)
        
        # Top-down view
        ax2 = fig.add_subplot(122)
        subsample = max(1, len(points) // 2000)
        ax2.scatter(points[::subsample, 0], points[::subsample, 1], 
                   c=point_colors[::subsample], s=2, alpha=0.6)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Top-down View')
        ax2.set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_mask_distribution(mask_logits, title="Mask Distribution", save_path=None, show=True):
    """
    Visualize the distribution of mask logits.
    
    Args:
        mask_logits: [K, N] mask logits
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    K, N = mask_logits.shape
    
    # Apply softmax to get probabilities
    mask_probs = np.exp(mask_logits) / np.exp(mask_logits).sum(axis=0, keepdims=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Histogram of max probabilities
    max_probs = mask_probs.max(axis=0)
    axes[0].hist(max_probs, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Max Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Max Probability Distribution\n(mean={max_probs.mean():.3f})')
    
    # 2. Points per segment
    seg_labels = mask_logits.argmax(axis=0)
    unique, counts = np.unique(seg_labels, return_counts=True)
    axes[1].bar(unique, counts, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Segment ID')
    axes[1].set_ylabel('Point Count')
    axes[1].set_title(f'Points per Segment\n({len(unique)} active segments)')
    
    # 3. Entropy of mask probabilities
    entropy = -np.sum(mask_probs * np.log(mask_probs + 1e-10), axis=0)
    axes[2].hist(entropy, bins=50, edgecolor='black', alpha=0.7)
    axes[2].axvline(np.log(K), color='r', linestyle='--', label=f'Max entropy (log {K})')
    axes[2].set_xlabel('Entropy')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Mask Entropy Distribution\n(mean={entropy.mean():.3f})')
    axes[2].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def load_and_visualize(npz_path, save_dir=None, show=True):
    """
    Load npz file and visualize.
    
    Args:
        npz_path: Path to the npz file
        save_dir: Directory to save figures (if None, saves next to npz)
        show: Whether to display plots
    """
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    
    points = data['points']
    seg_labels = data['seg_labels']
    mask_logits = data.get('mask_logits', None)
    flow = data.get('flow', None)
    
    print(f"  Points shape: {points.shape}")
    print(f"  Seg labels shape: {seg_labels.shape}")
    print(f"  Unique segments: {len(np.unique(seg_labels))}")
    if mask_logits is not None:
        print(f"  Mask logits shape: {mask_logits.shape}")
    if flow is not None:
        print(f"  Flow shape: {flow.shape}")
    
    # Determine save directory
    if save_dir is None:
        save_dir = os.path.dirname(npz_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Get step from path for title
    step_str = os.path.basename(os.path.dirname(npz_path))
    
    # Visualize segmentation
    seg_save_path = os.path.join(save_dir, "segmentation.png")
    visualize_segmentation(
        points, seg_labels, flow,
        title=f"Segmentation Results - {step_str}",
        save_path=seg_save_path,
        show=show
    )
    
    # Visualize mask distribution if available
    if mask_logits is not None:
        mask_save_path = os.path.join(save_dir, "mask_distribution.png")
        visualize_mask_distribution(
            mask_logits,
            title=f"Mask Distribution - {step_str}",
            save_path=mask_save_path,
            show=show
        )
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud segmentation results")
    parser.add_argument("--input", type=str, help="Path to npz file")
    parser.add_argument("--input_dir", type=str, help="Base output directory")
    parser.add_argument("--step", type=int, help="Step number (used with --input_dir)")
    parser.add_argument("--no_show", action="store_true", help="Don't display plots, just save")
    parser.add_argument("--all_steps", action="store_true", help="Visualize all steps in input_dir")
    args = parser.parse_args()
    
    show = not args.no_show
    
    if args.input:
        # Single file
        load_and_visualize(args.input, show=show)
    elif args.input_dir and args.step is not None:
        # Specific step
        npz_path = os.path.join(args.input_dir, f"step_{args.step:06d}", "sample.npz")
        if os.path.exists(npz_path):
            load_and_visualize(npz_path, show=show)
        else:
            print(f"File not found: {npz_path}")
    elif args.input_dir and args.all_steps:
        # All steps
        step_dirs = sorted([d for d in os.listdir(args.input_dir) if d.startswith("step_")])
        print(f"Found {len(step_dirs)} step directories")
        for step_dir in step_dirs:
            npz_path = os.path.join(args.input_dir, step_dir, "sample.npz")
            if os.path.exists(npz_path):
                print(f"\n{'='*50}")
                load_and_visualize(npz_path, show=False)  # Don't show for batch processing
        print(f"\nProcessed {len(step_dirs)} steps. Figures saved in respective directories.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

