"""
Model definitions for trajectories_for_seg.
"""

# from .sonata import SonataModel
# from .flow_head import FlowHead
# from .teacher_student import TeacherStudentModel
# from .segmentation_head import SegmentationHead
# from .segmentation_model import SegmentationModel
from .ogc_pointnet import OGCPointNetModel
# from .sparse_unet_seg import SparseUNetSegModel

__all__ = ['SonataModel', 'FlowHead', 'TeacherStudentModel', 'SegmentationHead', 'SegmentationModel', 'OGCPointNetModel', 'SparseUNetSegModel']

__all__ = ['OGCPointNetModel']
