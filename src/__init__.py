"""
RGB-Depth Frame Extractor
ROS2 bag에서 RGB-Depth 이미지 쌍을 추출하는 패키지
"""

__version__ = '2.0.0'
__author__ = 'Frame Extractor Team'

from .bag_reader import (
    RosBagReader,
    ImageMessage,
    ImuMessage,
    CameraInfoMessage,
    GpsMessage,
    MetadataMessage
)
from .frame_synchronizer import FrameSynchronizer, SynchronizedFrame
from .frame_extractor import FrameExtractor, load_depth_image, load_metadata
from .bag_validator import BagValidator

__all__ = [
    'RosBagReader',
    'ImageMessage',
    'ImuMessage',
    'CameraInfoMessage',
    'GpsMessage',
    'MetadataMessage',
    'FrameSynchronizer',
    'SynchronizedFrame',
    'FrameExtractor',
    'load_depth_image',
    'load_metadata',
    'BagValidator',
]
