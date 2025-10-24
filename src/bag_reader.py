"""
ROS Bag Reader Module
ROS2 bag 파일을 읽고 메시지를 추출하는 모듈
"""

import os
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, Set
from dataclasses import dataclass
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types


@dataclass
class ImageMessage:
    """Image message data class"""
    timestamp: int  # nanoseconds since epoch
    data: np.ndarray
    encoding: str
    width: int
    height: int


@dataclass
class ImuMessage:
    """IMU message data class"""
    timestamp: int  # nanoseconds since epoch
    angular_velocity: np.ndarray  # [x, y, z] rad/s
    linear_acceleration: np.ndarray  # [x, y, z] m/s^2


@dataclass
class CameraInfoMessage:
    """CameraInfo message data class"""
    timestamp: int  # nanoseconds since epoch
    width: int
    height: int
    K: np.ndarray  # 3x3 intrinsic matrix (flattened to 9 elements)
    D: np.ndarray  # Distortion coefficients
    R: np.ndarray  # 3x3 rectification matrix
    P: np.ndarray  # 3x4 projection matrix


@dataclass
class MetadataMessage:
    """RealSense Metadata message data class"""
    timestamp: int  # nanoseconds since epoch
    frame_timestamp: int
    frame_counter: int
    exposure_time: float  # microseconds
    gain: float
    brightness: float


@dataclass
class GpsMessage:
    """GPS message data class (Float64)"""
    timestamp: int  # nanoseconds since epoch
    value: float  # GPS degree value


@dataclass
class NavSatFixMessage:
    """GPS NavSatFix message data class (sensor_msgs/msg/NavSatFix)"""
    timestamp: int  # nanoseconds since epoch
    latitude: float  # Latitude in degrees
    longitude: float  # Longitude in degrees
    altitude: float  # Altitude in meters
    status: int  # GPS fix status: -1=No Fix, 0=Fix, 1=SBAS Fix, 2=GBAS Fix
    position_covariance_type: int  # Type of covariance: 0=Unknown, 1=Approximated, 2=Diagonal Known, 3=Known


class RosBagReader:
    """ROS2 bag 파일을 읽고 topic 데이터를 추출하는 클래스"""

    def __init__(self, bag_path: str, flip_rgb: bool = False, flip_depth: bool = False):
        """
        초기화

        Args:
            bag_path: ROS bag 파일 경로 (.bag directory)
            flip_rgb: RGB 이미지를 상하 반전할지 여부
            flip_depth: Depth 이미지를 상하 반전할지 여부
        """
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag path not found: {bag_path}")

        self.reader = None
        self.flip_rgb = flip_rgb
        self.flip_depth = flip_depth

    def __enter__(self):
        """Context manager entry"""
        self.reader = Reader(self.bag_path)
        self.reader.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.reader:
            self.reader.close()

    def get_topics(self) -> Dict[str, str]:
        """
        사용 가능한 모든 topic과 type을 반환

        Returns:
            topic 이름과 type의 dictionary
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        topics = {}
        for connection in self.reader.connections:
            topics[connection.topic] = connection.msgtype
        return topics

    def get_message_count(self, topic: str) -> int:
        """
        특정 topic의 메시지 개수 반환

        Args:
            topic: topic 이름

        Returns:
            메시지 개수
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        count = 0
        for connection, _, _ in self.reader.messages():
            if connection.topic == topic:
                count += 1
        return count

    def detect_topics(self, topic_type: str) -> list:
        """
        특정 타입의 사용 가능한 topic 목록 반환

        Args:
            topic_type: 'rgb', 'depth', 'imu' 등

        Returns:
            사용 가능한 topic 이름 리스트 (message count > 0인 것만)
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        topic_patterns = {
            'rgb': [
                '/camera/camera/color/image_raw/compressed',
                '/camera/camera/color/image_raw',
                '/camera/color/image_raw/compressed',
                '/camera/color/image_raw'
            ],
            'depth': [
                '/camera/camera/aligned_depth_to_color/image_raw',
                '/camera/camera/depth/image_raw',
                '/camera/depth/image_raw'
            ],
            'imu': [
                '/camera/camera/imu',
                '/camera/imu',
                '/imu'
            ]
        }

        if topic_type not in topic_patterns:
            return []

        available = []
        for pattern in topic_patterns[topic_type]:
            for conn in self.reader.connections:
                if conn.topic == pattern and conn.msgcount > 0:
                    available.append(conn.topic)
                    break  # 같은 패턴 중복 방지

        return available

    def read_images(self, topic: str, flip_vertical: bool = None) -> Iterator[ImageMessage]:
        """
        Image topic에서 이미지 메시지를 순차적으로 읽기

        자동으로 Image와 CompressedImage 타입 모두 지원

        Args:
            topic: image topic 이름
            flip_vertical: 이미지를 상하 반전할지 여부 (None이면 자동 판단)

        Yields:
            ImageMessage 객체
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        # Topic의 message type 확인
        connection = self._get_connection(topic)
        if not connection:
            raise ValueError(f"Topic not found: {topic}")

        msgtype = connection.msgtype

        # CompressedImage vs Image 판단
        if 'CompressedImage' in msgtype:
            yield from self._read_compressed_images(topic, flip_vertical)
        else:
            yield from self._read_raw_images(topic, flip_vertical)

    def _get_connection(self, topic: str):
        """Topic에 해당하는 connection 정보 반환"""
        for connection in self.reader.connections:
            if connection.topic == topic:
                return connection
        return None

    def _read_raw_images(self, topic: str, flip_vertical: bool = None) -> Iterator[ImageMessage]:
        """
        Raw Image 메시지 처리 (기존 로직)

        Args:
            topic: image topic 이름
            flip_vertical: 이미지를 상하 반전할지 여부

        Yields:
            ImageMessage 객체
        """
        for connection, timestamp, rawdata in self.reader.messages():
            if connection.topic == topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)

                # Image data를 numpy array로 변환
                if msg.encoding == 'rgb8':
                    # RGB8: 3 channels, uint8
                    image_data = np.frombuffer(msg.data, dtype=np.uint8)
                    image_data = image_data.reshape((msg.height, msg.width, 3))
                elif msg.encoding == 'bgr8':
                    # BGR8: 3 channels, uint8
                    image_data = np.frombuffer(msg.data, dtype=np.uint8)
                    image_data = image_data.reshape((msg.height, msg.width, 3))
                elif msg.encoding == '16UC1':
                    # 16-bit depth image
                    image_data = np.frombuffer(msg.data, dtype=np.uint16)
                    image_data = image_data.reshape((msg.height, msg.width))
                else:
                    # 기타 encoding은 raw data로 저장
                    image_data = np.frombuffer(msg.data, dtype=np.uint8)

                # Apply vertical flip if requested
                if flip_vertical or (flip_vertical is None and self._should_flip(msg.encoding)):
                    image_data = np.flipud(image_data)

                yield ImageMessage(
                    timestamp=timestamp,
                    data=image_data,
                    encoding=msg.encoding,
                    width=msg.width,
                    height=msg.height
                )

    def _read_compressed_images(self, topic: str, flip_vertical: bool = None) -> Iterator[ImageMessage]:
        """
        CompressedImage 메시지 처리 및 해제

        지원 포맷: JPEG, PNG

        Args:
            topic: compressed image topic 이름
            flip_vertical: 이미지를 상하 반전할지 여부

        Yields:
            ImageMessage 객체
        """
        for connection, timestamp, rawdata in self.reader.messages():
            if connection.topic == topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)

                # CompressedImage → numpy array
                np_arr = np.frombuffer(msg.data, np.uint8)
                image_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if image_data is None:
                    continue  # 디코딩 실패 시 skip

                # OpenCV는 BGR이므로 RGB로 변환
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

                # Flip 처리
                if flip_vertical or (flip_vertical is None and self.flip_rgb):
                    image_data = np.flipud(image_data)

                yield ImageMessage(
                    timestamp=timestamp,
                    data=image_data,
                    encoding=msg.format,  # 'jpeg', 'png' 등
                    width=image_data.shape[1],
                    height=image_data.shape[0]
                )

    def _should_flip(self, encoding: str) -> bool:
        """
        Determine if image should be flipped based on encoding type

        Args:
            encoding: Image encoding type

        Returns:
            True if should flip, False otherwise
        """
        if encoding in ['rgb8', 'bgr8']:
            return self.flip_rgb
        elif encoding == '16UC1':
            return self.flip_depth
        return False

    def read_imu(self, topic: str) -> Iterator[ImuMessage]:
        """
        IMU topic에서 IMU 메시지를 순차적으로 읽기

        Args:
            topic: IMU topic 이름

        Yields:
            ImuMessage 객체
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        for connection, timestamp, rawdata in self.reader.messages():
            if connection.topic == topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)

                angular_vel = np.array([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ])

                linear_acc = np.array([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ])

                yield ImuMessage(
                    timestamp=timestamp,
                    angular_velocity=angular_vel,
                    linear_acceleration=linear_acc
                )

    def get_time_range(self) -> Tuple[int, int]:
        """
        Bag 파일의 시작 및 종료 timestamp 반환

        Returns:
            (start_time, end_time) in nanoseconds
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        return (self.reader.start_time, self.reader.end_time)

    def get_duration_seconds(self) -> float:
        """
        Bag 파일의 전체 duration을 초 단위로 반환

        Returns:
            duration in seconds
        """
        start, end = self.get_time_range()
        return (end - start) / 1e9

    def read_multiple_topics(
        self,
        rgb_topic: str,
        depth_topic: str,
        imu_topic: str,
        lazy_load_images: bool = True
    ) -> Tuple[Dict, Dict, Dict]:
        """
        한 번의 순회로 여러 topic의 데이터를 동시에 읽기 (성능 최적화)

        Args:
            rgb_topic: RGB 이미지 topic 이름
            depth_topic: Depth 이미지 topic 이름
            imu_topic: IMU topic 이름
            lazy_load_images: True면 timestamp만 수집, False면 이미지도 로드

        Returns:
            (rgb_data, depth_data, imu_data) tuple
            - rgb_data: {'timestamps': [...], 'frames': [...] or None}
            - depth_data: {'timestamps': [...], 'frames': [...] or None}
            - imu_data: {'timestamps': [...], 'angular_velocities': [...]}
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        # 데이터 버퍼 초기화
        rgb_data = {'timestamps': [], 'frames': [] if not lazy_load_images else None}
        depth_data = {'timestamps': [], 'frames': [] if not lazy_load_images else None}
        imu_data = {'timestamps': [], 'angular_velocities': []}

        # 단일 순회로 모든 topic 데이터 수집
        for connection, timestamp, rawdata in self.reader.messages():
            if connection.topic == rgb_topic:
                rgb_data['timestamps'].append(timestamp)
                if not lazy_load_images:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    image_data = self._decode_image(msg)
                    rgb_data['frames'].append(image_data)

            elif connection.topic == depth_topic:
                depth_data['timestamps'].append(timestamp)
                if not lazy_load_images:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    image_data = self._decode_image(msg)
                    depth_data['frames'].append(image_data)

            elif connection.topic == imu_topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                angular_vel = np.array([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ])
                imu_data['timestamps'].append(timestamp)
                imu_data['angular_velocities'].append(angular_vel)

        # NumPy 배열로 변환
        rgb_data['timestamps'] = np.array(rgb_data['timestamps'])
        depth_data['timestamps'] = np.array(depth_data['timestamps'])
        imu_data['timestamps'] = np.array(imu_data['timestamps'])
        imu_data['angular_velocities'] = np.array(imu_data['angular_velocities'])

        return rgb_data, depth_data, imu_data

    def _decode_image(self, msg, flip_vertical: bool = None, is_compressed: bool = False) -> np.ndarray:
        """
        ROS 이미지 메시지를 NumPy 배열로 디코딩

        자동으로 Image와 CompressedImage 타입 모두 지원

        Args:
            msg: ROS Image or CompressedImage message
            flip_vertical: 이미지를 상하 반전할지 여부
            is_compressed: CompressedImage 여부 (자동 감지 가능)

        Returns:
            NumPy array
        """
        # CompressedImage 감지 및 처리
        if is_compressed or hasattr(msg, 'format'):
            # CompressedImage 디코딩
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image_data is None:
                raise ValueError("Failed to decode compressed image")

            # OpenCV는 BGR이므로 RGB로 변환
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

            # Flip 처리
            if flip_vertical or (flip_vertical is None and self.flip_rgb):
                image_data = np.flipud(image_data)

        else:
            # Raw Image 디코딩
            if msg.encoding == 'rgb8':
                image_data = np.frombuffer(msg.data, dtype=np.uint8)
                image_data = image_data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'bgr8':
                image_data = np.frombuffer(msg.data, dtype=np.uint8)
                image_data = image_data.reshape((msg.height, msg.width, 3))
            elif msg.encoding == '16UC1':
                image_data = np.frombuffer(msg.data, dtype=np.uint16)
                image_data = image_data.reshape((msg.height, msg.width))
            else:
                image_data = np.frombuffer(msg.data, dtype=np.uint8)

            # Apply vertical flip if requested
            if flip_vertical or (flip_vertical is None and self._should_flip(msg.encoding)):
                image_data = np.flipud(image_data)

        return image_data

    def read_images_by_indices(
        self,
        topic: str,
        target_timestamps: np.ndarray,
        flip_vertical: bool = None
    ) -> Dict[int, np.ndarray]:
        """
        특정 timestamp의 이미지만 선택적으로 로드 (Lazy Loading 지원)

        자동으로 Image와 CompressedImage 타입 모두 지원

        Args:
            topic: image topic 이름
            target_timestamps: 로드할 timestamp 배열 (정렬된 상태)
            flip_vertical: 이미지를 상하 반전할지 여부 (None이면 자동 판단)

        Returns:
            {timestamp: image_data} dictionary
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        # Topic의 message type 확인
        connection = self._get_connection(topic)
        if not connection:
            raise ValueError(f"Topic not found: {topic}")

        is_compressed = 'CompressedImage' in connection.msgtype

        target_set = set(target_timestamps)
        images = {}

        for connection, timestamp, rawdata in self.reader.messages():
            if connection.topic == topic and timestamp in target_set:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                image_data = self._decode_image(msg, flip_vertical, is_compressed)
                images[timestamp] = image_data

                # 모든 타겟을 찾았으면 조기 종료
                if len(images) == len(target_set):
                    break

        return images

    def read_camera_info(self, topic: str) -> Iterator[CameraInfoMessage]:
        """
        CameraInfo topic에서 카메라 파라미터 읽기

        Args:
            topic: CameraInfo topic 이름

        Yields:
            CameraInfoMessage 객체
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        for connection, timestamp, rawdata in self.reader.messages():
            if connection.topic == topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)

                yield CameraInfoMessage(
                    timestamp=timestamp,
                    width=msg.width,
                    height=msg.height,
                    K=np.array(msg.k),  # 3x3 matrix flattened to 9 elements
                    D=np.array(msg.d),  # Distortion coefficients
                    R=np.array(msg.r),  # 3x3 rectification matrix
                    P=np.array(msg.p)   # 3x4 projection matrix
                )

    def read_gps(self, topic: str) -> Iterator[GpsMessage]:
        """
        GPS topic에서 GPS 데이터 읽기 (Float64 타입)

        Args:
            topic: GPS topic 이름

        Yields:
            GpsMessage 객체
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        for connection, timestamp, rawdata in self.reader.messages():
            if connection.topic == topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)

                yield GpsMessage(
                    timestamp=timestamp,
                    value=msg.data
                )

    def read_navsatfix(self, topic: str) -> Iterator[NavSatFixMessage]:
        """
        GPS NavSatFix topic에서 GPS 데이터 읽기 (sensor_msgs/msg/NavSatFix 타입)

        Args:
            topic: GPS NavSatFix topic 이름 (예: '/gps/fix')

        Yields:
            NavSatFixMessage 객체
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        for connection, timestamp, rawdata in self.reader.messages():
            if connection.topic == topic:
                msg = deserialize_cdr(rawdata, connection.msgtype)

                yield NavSatFixMessage(
                    timestamp=timestamp,
                    latitude=msg.latitude,
                    longitude=msg.longitude,
                    altitude=msg.altitude,
                    status=msg.status.status,
                    position_covariance_type=msg.position_covariance_type
                )

    def read_metadata(self, topic: str) -> Iterator[MetadataMessage]:
        """
        RealSense Metadata topic에서 메타데이터 읽기

        Args:
            topic: Metadata topic 이름

        Yields:
            MetadataMessage 객체
        """
        if not self.reader:
            raise RuntimeError("Reader not opened. Use 'with' statement.")

        for connection, timestamp, rawdata in self.reader.messages():
            if connection.topic == topic:
                try:
                    msg = deserialize_cdr(rawdata, connection.msgtype)

                    # RealSense Metadata 구조 파싱
                    # json_data 필드에서 필요한 정보 추출
                    metadata_items = {}
                    if hasattr(msg, 'json_data'):
                        import json
                        try:
                            metadata_dict = json.loads(msg.json_data)
                            for item in metadata_dict:
                                metadata_items[item.get('key', '')] = item.get('value', 0)
                        except:
                            pass

                    yield MetadataMessage(
                        timestamp=timestamp,
                        frame_timestamp=metadata_items.get('frame_timestamp', 0),
                        frame_counter=metadata_items.get('frame_counter', 0),
                        exposure_time=metadata_items.get('exposure_time', 0.0),
                        gain=metadata_items.get('gain', 0.0),
                        brightness=metadata_items.get('brightness', 0.0)
                    )
                except Exception as e:
                    # Metadata 파싱 실패 시 기본값 반환
                    yield MetadataMessage(
                        timestamp=timestamp,
                        frame_timestamp=0,
                        frame_counter=0,
                        exposure_time=0.0,
                        gain=0.0,
                        brightness=0.0
                    )
