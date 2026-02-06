import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from datasets.seg_dataset import SegTrainDataset, SegValDataset


def visualize_bev_train(points, flow, save_path=None):
    """
    绘制训练集的BEV图，包含场景流
    
    Args:
        points: [N, 3] 点云坐标
        flow: [N, 3] 场景流向量
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 转换为numpy
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    
    # BEV: 使用x和y坐标
    x = points[:, 0]
    y = points[:, 1]
    
    # 绘制点云（BEV）
    ax.scatter(x, y, c='blue', s=0.5, alpha=0.6, label='Point Cloud')
    
    # 绘制场景流向量
    # 只绘制部分向量以避免过于密集
    step = max(1, len(points) // 1000)  # 最多绘制1000个向量
    x_flow = x[::step]
    y_flow = y[::step]
    flow_x = flow[::step, 0]
    flow_y = flow[::step, 1]
    
    # 归一化流向量以便可视化
    flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
    # max_magnitude = np.percentile(flow_magnitude, 95)  # 使用95分位数避免异常值
    # if max_magnitude > 0:
    #     scale = 1.0 / max_magnitude
    #     flow_x_scaled = flow_x * scale * 2.0  # 放大2倍以便看清
    #     flow_y_scaled = flow_y * scale * 2.0
    # else:
    flow_x_scaled = flow_x
    flow_y_scaled = flow_y
    
    # 绘制流向量（使用颜色表示大小）
    quiver = ax.quiver(x_flow, y_flow, flow_x_scaled, flow_y_scaled, 
                       flow_magnitude, cmap='hot', scale=20, width=0.002, 
                       alpha=0.7, label='Scene Flow')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Train Dataset - BEV with Scene Flow', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 添加颜色条
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label('Flow Magnitude', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved train BEV visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_bev_val(points, instance_label, save_path=None):
    """
    绘制验证集的BEV图，根据分割标签上色
    
    Args:
        points: [N, 3] 点云坐标
        instance_label: [N] 实例分割标签
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 转换为numpy
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(instance_label, torch.Tensor):
        instance_label = instance_label.cpu().numpy()
    
    # BEV: 使用x和y坐标
    x = points[:, 0]
    y = points[:, 1]
    
    # 获取唯一的标签
    unique_labels = np.unique(instance_label)
    num_labels = len(unique_labels)
    
    # 生成颜色映射
    if num_labels <= 20:
        # 使用tab20颜色映射
        cmap = plt.cm.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, num_labels))
    else:
        # 使用更多颜色
        cmap = plt.cm.get_cmap('nipy_spectral')
        colors = cmap(np.linspace(0, 1, num_labels))
    
    # 为每个标签创建颜色映射
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    point_colors = np.array([label_to_color[label] for label in instance_label])
    
    # 绘制点云（BEV），根据分割标签上色
    scatter = ax.scatter(x, y, c=point_colors, s=0.5, alpha=0.8)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Val Dataset - BEV with Segmentation ({num_labels} instances)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 添加图例（只显示前20个标签以避免过于拥挤）
    if num_labels <= 20:
        legend_elements = [mpatches.Patch(facecolor=colors[i], 
                                         label=f'Instance {unique_labels[i]}')
                          for i in range(num_labels)]
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=8, ncol=2, framealpha=0.7)
    else:
        ax.text(0.02, 0.98, f'{num_labels} instances', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved val BEV visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    # 从配置文件读取数据路径（或使用默认路径）
    train_dir = "/root/autodl-tmp/av2_seg/seg_train"
    val_dir = "/root/autodl-tmp/av2_seg/seg_val"
    
    # 创建输出目录
    output_dir = "bev_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Loading Train Dataset...")
    print("=" * 60)
    
    # 加载训练集
    train_dataset = SegTrainDataset(train_dir, num_points=32768)
    print(f"Train dataset size: {len(train_dataset)}")
    
    # 可视化训练集的几个样本
    num_train_samples = min(3, len(train_dataset))
    for i in range(num_train_samples):
        print(f"\nProcessing train sample {i+1}/{num_train_samples}...")
        sample = train_dataset[i]
        points = sample['pc']
        flow = sample['flow']
        
        save_path = os.path.join(output_dir, f"train_bev_sample_{i+1}.png")
        visualize_bev_train(points, flow, save_path=save_path)
    
    print("\n" + "=" * 60)
    print("Loading Val Dataset...")
    print("=" * 60)
    
    # 加载验证集
    val_dataset = SegValDataset(val_dir, num_points=32768)
    print(f"Val dataset size: {len(val_dataset)}")
    
    # 可视化验证集的几个样本
    num_val_samples = min(3, len(val_dataset))
    for i in range(num_val_samples):
        print(f"\nProcessing val sample {i+1}/{num_val_samples}...")
        sample = val_dataset[i]
        points = sample['pc']
        instance_label = sample['instance_label']
        
        save_path = os.path.join(output_dir, f"val_bev_sample_{i+1}.png")
        visualize_bev_val(points, instance_label, save_path=save_path)
    
    print("\n" + "=" * 60)
    print(f"Visualization complete! Results saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
