"""
PoseSense - Live Human Action Detection

A computer vision application for real-time human action recognition
using pose estimation and LSTM neural networks.
"""

__version__ = "1.0.0"
__author__ = "Prith-ML"
__description__ = "Live Human Action Detection using Pose Estimation and LSTM"

from .core.liveApplicationCode import SkeletonLSTM
from .core.config import *

__all__ = [
    'SkeletonLSTM',
    'MODEL_CONFIG',
    'POSE_CONFIG',
    'VISUALIZATION_CONFIG',
    'ACTION_LABELS',
    'LABEL_COLORS',
    'CAMERA_CONFIG',
    'PERFORMANCE_CONFIG'
] 