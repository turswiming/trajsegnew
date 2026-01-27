# Trajectories for Seg

A self-supervised learning project for scene flow estimation using Sonata (Point Transformer V3) architecture with teacher-student training paradigm.

## Features

- **Sonata Architecture**: Based on Point Transformer V3 for point cloud feature extraction
- **Teacher-Student Training**: EMA-based teacher model for self-supervised learning
- **Multiple Loss Functions**: FlowSmoothLoss, InvarianceLoss, and PointSmoothLoss (migrated from gan_seg)
- **Complete Training Pipeline**: Training, evaluation, and logging utilities
- **Independent Project**: Fully self-contained, can run independently

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Note: For pytorch3d, you may need to install from source or use conda:
# conda install -c fvcore -c pytorch3d pytorch3d
```

## Usage

### Training

```bash
# Train with default config
python train.py --config src/configs/config.yaml

# Or specify custom config
python train.py --config path/to/your/config.yaml
```

### Evaluation

```bash
# Evaluate a checkpoint
python eval.py --checkpoint checkpoints/best_epoch_10.pth --data_dir data/val
```

### Test Imports

```bash
# Verify all modules can be imported
python test_imports.py
```

## Project Structure

```
trajectories_for_seg/
├── src/
│   ├── models/              # Model definitions
│   │   ├── sonata.py        # Sonata (Point Transformer V3) wrapper
│   │   ├── flow_head.py     # Scene flow prediction head
│   │   └── teacher_student.py  # Teacher-Student training architecture
│   ├── losses/              # Loss functions (migrated from gan_seg)
│   │   ├── flow_smooth_loss.py
│   │   ├── invariance_loss.py
│   │   └── point_smooth_loss.py
│   ├── datasets/            # Data loaders
│   │   └── sceneflow_dataset.py
│   ├── utils/               # Utilities
│   │   ├── logger.py        # TensorBoard logging
│   │   └── metrics.py       # Evaluation metrics
│   └── configs/             # Configuration files
│       └── config.yaml
├── train.py                 # Training script
├── eval.py                  # Evaluation script
├── test_imports.py          # Import test script
├── requirements.txt         # Dependencies
└── README.md
```

## Configuration

Edit `src/configs/config.yaml` to customize:
- Model settings (pretrained weights, feature dimensions)
- Training hyperparameters (learning rate, batch size, epochs)
- Loss weights (consistency, flow smooth, invariance, point smooth)
- Data paths

## Teacher-Student Training

The project implements Sonata's special training approach:
1. **Student Model**: Trainable Sonata encoder + flow head
2. **Teacher Model**: EMA-updated copy of student (frozen)
3. **Training Process**:
   - Student generates predictions
   - Teacher generates pseudo-labels (no gradients)
   - Consistency loss aligns student with teacher
   - Self-supervised losses (FlowSmooth, Invariance, PointSmooth) provide additional supervision
   - Teacher updated via EMA after each step

## Loss Functions

- **FlowSmoothLoss**: Encourages smooth flow fields within segmented regions using quadratic approximation
- **InvarianceLoss**: Ensures segmentation consistency across different views
- **PointSmoothLoss**: Combines KNN and ball query approaches for spatially coherent segmentation

## Data Format

Expected data format (`.npz` files):
- `pc0`: [N, 3] first frame point cloud
- `pc1`: [N, 3] second frame point cloud  
- `flow`: [N, 3] ground truth flow (optional)
- `pose0`: [4, 4] first frame pose (optional)
- `pose1`: [4, 4] second frame pose (optional)

## Notes

- The project uses native PyTorch (not Lightning) to have full control over teacher-student EMA updates
- Sonata pre-trained weights are automatically downloaded from HuggingFace on first use
- All loss functions are migrated from the gan_seg repository and adapted for this project

## License

MIT License

