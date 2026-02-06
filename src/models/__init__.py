"""
Model definitions for trajectories_for_seg.
"""

# from .sonata import SonataModel
# from .flow_head import FlowHead
# from .teacher_student import TeacherStudentModel
# from .segmentation_head import SegmentationHead
# from .segmentation_model import SegmentationModel
# from .OGCModel.segnet_av2 import MaskFormer3D
# from .sparse_unet_seg import SparseUNetSegModel
from .deltaflow_seg import DeltaFlowSeg

__all__ = ['MaskFormer3D', 'SparseUNetSegModel', 'DeltaFlowSeg']
