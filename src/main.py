"""
Main Execution Script V2 (메인 실행 스크립트 V2)
CameraInfo, GPS, RealSense Metadata, IMU 통합을 통한 향상된 RGB-Depth frame 추출

주요 기능:
- 모든 동기화된 RGB-Depth 프레임 추출 (필터링 없음)
- Camera calibration (카메라 보정) 정보 추출
- RealSense Metadata (frame_counter, exposure_time, gain, brightness) 수집
- GPS metadata 통합
- IMU linear acceleration (x, y, z, magnitude) 수집
- 통합 metadata_all.csv 출력 (모든 센서 데이터 포함)
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from bag_reader import RosBagReader, ImageMessage, ImuMessage, CameraInfoMessage, GpsMessage, MetadataMessage, NavSatFixMessage
from frame_synchronizer import FrameSynchronizer, SynchronizedFrame
from frame_extractor import FrameExtractor
from bag_validator import BagValidator
from utils import find_closest_scalar, find_closest_dict, extract_dict_values
from gps_processor import GPSProcessor
from sensor_extractor import SensorExtractor
from metadata_writer import MetadataWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RgbDepthExtractorPipelineV2:
    """
    향상된 RGB-Depth Frame 추출 Pipeline V2
    Enhanced RGB-Depth Frame Extraction Pipeline V2

    주요 기능 (Key Features):
    - 모든 동기화된 RGB-Depth 프레임 추출 (필터링 없음)
    - CameraInfo 통합 (camera intrinsics 포함)
    - RealSense Metadata 수집 (frame_counter, exposure_time, gain, brightness)
    - GPS metadata 통합
    - IMU linear acceleration 수집 (x, y, z, magnitude)
    - 통합 metadata_all.csv 출력 (15개 필드)

    처리 단계 (Processing Steps):
    1. ROS bag validation (bag_validator.py)
    2. Camera calibration 정보 추출 (optional)
    3. IMU acceleration 데이터 추출 (metadata용)
    4. RealSense metadata 추출
    5. RGB/Depth 이미지 동기화 (모든 프레임)
    6. GPS metadata 추출
    7. 향상된 metadata와 함께 파일 저장
    8. Summary 및 통계 출력
    """

    # 디스플레이 상수 (Display Constants)
    HEADER_WIDTH = 60

    # 시간 변환 상수 (Time Conversion Constants)
    NS_TO_MS = 1e6  # nanoseconds → milliseconds
    NS_TO_S = 1e9  # nanoseconds → seconds
    MS_TO_NS = 1e6  # milliseconds → nanoseconds

    # Frame ID 포맷 (Frame ID Format)
    FRAME_ID_FORMAT = "frame_{:06d}"  # frame_index

    # 기본 설정 상수 (Default Configuration Constants)
    DEFAULT_TIME_TOLERANCE_MS = 33  # RGB/Depth 동기화 허용 오차 (milliseconds)
    DEFAULT_PARALLEL_WORKERS = 4  # 병렬 처리 worker 수
    MIN_PARALLEL_WORKERS = 1  # 최소 worker 수

    def __init__(
        self,
        bag_path: str,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Pipeline 초기화
        Initialize pipeline

        Args:
            bag_path: ROS bag 파일 경로
            output_dir: 출력 디렉토리 경로
            config: 설정 dictionary
        """
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)

        # 향상된 기본 설정 (Enhanced default configuration)
        default_config = {
            # 이미지 topic (Image topics)
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'imu_topic': '/camera/camera/imu',  # RealSense 내장 IMU (실제 가속도 데이터 포함)

            # 추가 topic (Additional topics)
            'rgb_camera_info_topic': '/camera/camera/color/camera_info',
            'depth_camera_info_topic': '/camera/camera/aligned_depth_to_color/camera_info',
            'metadata_topic': '/camera/camera/color/metadata',
            'gps_topic': '/gps/fix',

            # 동기화 (Synchronization)
            'time_tolerance_ms': self.DEFAULT_TIME_TOLERANCE_MS,

            # 출력 포맷 (Output format)
            'rgb_format': 'png',
            'depth_format': 'png',
            'save_visualization': True,

            # 성능 최적화 (Performance - 강제 활성화)
            'flip_rgb': True,  # RGB 이미지 상하 반전
            'flip_depth': True,  # Depth 이미지 상하 반전
            'optimized': True,  # 최적화 모드 (lazy loading)
            'parallel_workers': self.DEFAULT_PARALLEL_WORKERS,

            # 새로운 기능 (New features - 선택적)
            'save_camera_info': False,  # Camera calibration 저장
            'save_gps_metadata': True,  # GPS 데이터 저장
        }

        self.config = {**default_config, **(config or {})}

        # parallel_workers 최소값 보장 (Ensure minimum parallel workers)
        if self.config['parallel_workers'] < self.MIN_PARALLEL_WORKERS:
            self.config['parallel_workers'] = self.DEFAULT_PARALLEL_WORKERS

        # Component 초기화 (Initialize components)
        self.synchronizer = FrameSynchronizer(
            time_tolerance_ns=int(self.config['time_tolerance_ms'] * self.MS_TO_NS)
        )

        self.extractor = FrameExtractor(
            output_dir=output_dir,
            rgb_format=self.config['rgb_format'],
            depth_format=self.config['depth_format'],
            save_visualization=self.config['save_visualization']
        )

        # 추가 metadata 저장소 (Storage for additional metadata)
        self.camera_calibration = {}  # Camera calibration 정보
        self.all_metadata_records = []  # 모든 frame의 metadata 기록

        self.gps_processor = GPSProcessor(self.logger, self.config)
        self.sensor_extractor = SensorExtractor(self.logger, self.config, self.camera_calibration)
        self.metadata_writer = MetadataWriter(
            self.logger,
            self.config,
            self.extractor,
            self.output_dir,
            self.all_metadata_records,
            self.camera_calibration,
            self.bag_path
        )

    def _print_pipeline_header(self) -> None:
        """
        Pipeline 헤더 정보 출력
        Print pipeline header information
        """
        self.logger.info("=" * self.HEADER_WIDTH)
        self.logger.info("ROS Bag RGB-Depth Frame Extractor V2")
        self.logger.info("=" * self.HEADER_WIDTH)
        self.logger.info(f"Bag path: {self.bag_path}")
        self.logger.info(f"Output path: {self.output_dir}")
        self.logger.info(f"Optimized mode: {self.config['optimized']}")
        self.logger.info(f"Parallel workers: {self.config['parallel_workers']}")
        self.logger.info(f"Flip images: RGB={self.config['flip_rgb']}, Depth={self.config['flip_depth']}")
        self.logger.info("=" * self.HEADER_WIDTH)

    def _print_pipeline_summary(self, total_extracted: int) -> None:
        """
        Pipeline 완료 요약 출력
        Print pipeline completion summary
        """
        self.logger.info("=" * self.HEADER_WIDTH)
        self.logger.info("Extraction completed!")
        self.logger.info(f"  Total extracted frames: {total_extracted}")
        self.logger.info(f"  Output path: {self.output_dir}")
        self.logger.info("=" * self.HEADER_WIDTH)

    def _process_camera_info(self, reader: RosBagReader) -> None:
        """
        Camera calibration 정보 처리
        Process camera calibration information
        """
        if self.config['save_camera_info']:
            self.logger.info("[1/6] Extracting camera calibration...")
            self.sensor_extractor.extract_camera_info(reader)

    def _process_images(self, reader: RosBagReader) -> Tuple[List[SynchronizedFrame], Dict[str, Any]]:
        """
        RGB/Depth 이미지 처리 및 동기화
        Process and synchronize RGB/Depth images

        Returns:
            (동기화된 frame 리스트, 동기화 통계 dictionary)
        """
        self.logger.info("[4/6] Reading and synchronizing frames (optimized)...")
        all_synchronized_frames = self._read_and_synchronize_images_optimized(reader)

        self.logger.info(f"  Final frame count: {len(all_synchronized_frames)}")

        if len(all_synchronized_frames) > 0:
            sync_stats = self.synchronizer.get_synchronization_statistics(
                all_synchronized_frames
            )
            self.logger.info(f"  Average time difference: {sync_stats['mean_time_diff_ms']:.2f}ms")
            self.logger.info(f"  Maximum time difference: {sync_stats['max_time_diff_ms']:.2f}ms")
        else:
            sync_stats = {'total_pairs': 0}

        return all_synchronized_frames, sync_stats

    def _process_additional_metadata(self, reader: RosBagReader) -> Dict[int, float]:
        """
        GPS metadata 처리
        Process GPS metadata

        Returns:
            GPS 데이터 dictionary
        """
        self.logger.info("[5/6] Extracting GPS metadata...")
        gps_data = self.gps_processor.extract_gps_data(reader) if self.config['save_gps_metadata'] else {}
        return gps_data

    def _save_results(
        self,
        all_synchronized_frames: List[SynchronizedFrame],
        imu_data: Dict[int, Dict[str, float]],
        rs_metadata: Dict[int, Dict[str, Any]],
        gps_data: Dict[int, float],
        sync_stats: Dict[str, Any]
    ) -> int:
        """Save all frames and metadata"""
        self.logger.info("[6/6] Saving frames...")
        total_extracted = self.metadata_writer.save_all_frames(
            all_synchronized_frames,
            imu_data,
            rs_metadata,
            gps_data
        )

        self.logger.info("[7/7] Saving metadata...")
        self.metadata_writer.save_unified_metadata()
        self.metadata_writer.save_camera_calibration()
        self.metadata_writer.save_extraction_summary(sync_stats, total_extracted)

        return total_extracted

    def run(self) -> None:
        """
        전체 향상된 pipeline 실행
        Execute the entire enhanced pipeline
        """
        self._print_pipeline_header()

        # 0. Bag 검증 (Validate bag metadata)
        self.logger.info("=" * self.HEADER_WIDTH)
        self.logger.info("Validating bag metadata...")
        self.logger.info("=" * self.HEADER_WIDTH)

        validator = BagValidator()
        validation_result = validator.validate_bag(self.bag_path)

        # 검증 결과 출력
        validator.print_validation_result(validation_result)

        # 검증 결과를 log 파일로 저장 (Save validation result to log file)
        bag_path_obj = Path(self.bag_path)
        log_path = bag_path_obj / f"{bag_path_obj.name}.log"
        validator.save_validation_log(validation_result, str(log_path))
        self.logger.info(f"Validation log saved to: {log_path}")

        # 검증 실패 시 중단
        if not validation_result.is_valid:
            self.logger.error("Bag validation failed. Aborting extraction.")
            return

        # 경고가 있으면 알림
        if validation_result.has_warnings():
            self.logger.warning("Bag validation passed with warnings. Proceeding with extraction...")

        self.logger.info("=" * self.HEADER_WIDTH)

        with RosBagReader(
            self.bag_path,
            flip_rgb=self.config['flip_rgb'],
            flip_depth=self.config['flip_depth']
        ) as reader:
            # 0. Topic 자동 탐지 (Auto-detect topics)
            self._auto_detect_topics(reader)

            # 1. Topic 정보 표시 (Display topic information)
            self._log_bag_info(reader)

            # 2. CameraInfo 추출 (Extract CameraInfo)
            self._process_camera_info(reader)

            # 3. IMU 가속도 데이터 추출 (Extract IMU acceleration data)
            imu_data = self.sensor_extractor.extract_imu_acceleration(reader)

            # 4. RealSense Metadata 추출 (Extract RealSense metadata)
            # Skip RealSense metadata - user only needs RGB-D, IMU, GPS, timestamp from bag
            rs_metadata = {}

            # 5. 이미지 처리 및 동기화 (Process and synchronize images)
            all_synchronized_frames, sync_stats = self._process_images(reader)

            # 6. GPS metadata 추출 (Extract GPS metadata)
            gps_data = self._process_additional_metadata(reader)

            # 7. 모든 결과 저장 (Save all results)
            total_extracted = self._save_results(
                all_synchronized_frames,
                imu_data,
                rs_metadata,
                gps_data,
                sync_stats
            )

            # 8. 요약 출력 (Print summary)
            self._print_pipeline_summary(total_extracted)

    def _log_bag_info(self, reader: RosBagReader) -> None:
        """Log bag information"""
        topics = reader.get_topics()
        duration = reader.get_duration_seconds()

        self.logger.info("Bag information:")
        self.logger.info(f"  Total duration: {duration:.1f}s")
        self.logger.info(f"  Total topics: {len(topics)}")
        self.logger.info("Main topics:")
        for topic_name in [
            self.config['rgb_topic'],
            self.config['depth_topic'],
            self.config['imu_topic'],
            self.config.get('rgb_camera_info_topic', ''),
            self.config.get('gps_topic', '')
        ]:
            if topic_name and topic_name in topics:
                self.logger.info(f"    {topic_name}: {topics[topic_name]}")

    def _auto_detect_topics(self, reader: RosBagReader) -> None:
        """
        Bag 파일에서 사용 가능한 topic 자동 탐지 및 설정

        우선순위:
        1. CLI 인자로 명시된 topic (최우선)
        2. 자동 탐지된 topic
        3. 기본값
        """
        # RGB topic
        if not self._topic_exists(reader, self.config['rgb_topic']):
            self.logger.info(f"  RGB topic '{self.config['rgb_topic']}' not found, auto-detecting...")
            rgb_candidates = reader.detect_topics('rgb')
            if rgb_candidates:
                self.config['rgb_topic'] = rgb_candidates[0]
                topic_type = 'CompressedImage' if 'compressed' in rgb_candidates[0] else 'Raw Image'
                self.logger.info(f"  → Using: {self.config['rgb_topic']} ({topic_type})")
            else:
                self.logger.warning("  → No RGB topic found!")

        # Depth topic
        if not self._topic_exists(reader, self.config['depth_topic']):
            self.logger.info(f"  Depth topic '{self.config['depth_topic']}' not found, auto-detecting...")
            depth_candidates = reader.detect_topics('depth')
            if depth_candidates:
                self.config['depth_topic'] = depth_candidates[0]
                self.logger.info(f"  → Using: {self.config['depth_topic']}")
            else:
                self.logger.warning("  → No Depth topic found!")

        # IMU topic
        if not self._topic_exists(reader, self.config['imu_topic']):
            self.logger.info(f"  IMU topic '{self.config['imu_topic']}' not found, auto-detecting...")
            imu_candidates = reader.detect_topics('imu')
            if imu_candidates:
                self.config['imu_topic'] = imu_candidates[0]
                self.logger.info(f"  → Using: {self.config['imu_topic']}")
            else:
                self.logger.warning("  → No IMU topic found!")

    def _topic_exists(self, reader: RosBagReader, topic: str) -> bool:
        """Topic이 존재하고 메시지가 있는지 확인"""
        for conn in reader.reader.connections:
            if conn.topic == topic and conn.msgcount > 0:
                return True
        return False

    def _read_and_synchronize_images_optimized(
        self,
        reader: RosBagReader
    ) -> List[SynchronizedFrame]:
        """
        Optimized image reading and synchronization (all frames, no filtering)
        """
        self.logger.info("  Optimized single-pass reading...")

        # Step 1: Collect timestamps (lazy loading)
        rgb_data, depth_data, _ = reader.read_multiple_topics(
            self.config['rgb_topic'],
            self.config['depth_topic'],
            self.config['imu_topic'],
            lazy_load_images=True
        )

        self.logger.info(f"    RGB: {len(rgb_data['timestamps'])} timestamps")
        self.logger.info(f"    Depth: {len(depth_data['timestamps'])} timestamps")

        # Step 2: Fast synchronization
        self.logger.info("  Fast synchronization...")
        synchronized_metadata = self.synchronizer.synchronize_frames_fast(
            rgb_data['timestamps'],
            None,
            depth_data['timestamps'],
            None
        )

        self.logger.info(f"    Synchronized pairs: {len(synchronized_metadata)}")

        # Step 3: Load all synchronized frames
        self.logger.info(f"  Loading all synchronized frames...")

        rgb_timestamps_to_load = np.array([f.rgb_timestamp for f in synchronized_metadata])
        depth_timestamps_to_load = np.array([f.depth_timestamp for f in synchronized_metadata])

        rgb_images = reader.read_images_by_indices(
            self.config['rgb_topic'],
            rgb_timestamps_to_load
        )
        depth_images = reader.read_images_by_indices(
            self.config['depth_topic'],
            depth_timestamps_to_load
        )

        self.logger.info(f"    Loaded RGB: {len(rgb_images)}")
        self.logger.info(f"    Loaded Depth: {len(depth_images)}")

        # Step 4: Combine metadata with image data
        synchronized_frames = []
        for meta in tqdm(synchronized_metadata, desc="  Combining frames"):
            rgb_img = rgb_images.get(meta.rgb_timestamp)
            depth_img = depth_images.get(meta.depth_timestamp)

            if rgb_img is not None and depth_img is not None:
                synchronized_frames.append(SynchronizedFrame(
                    rgb_timestamp=meta.rgb_timestamp,
                    depth_timestamp=meta.depth_timestamp,
                    rgb_data=rgb_img,
                    depth_data=depth_img,
                    time_diff=meta.time_diff
                ))

        return synchronized_frames


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Extract RGB-Depth frames with enhanced metadata (V2)'
    )

    parser.add_argument(
        'bag_path',
        type=str,
        help='ROS bag file path (.bag directory)'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory path'
    )

    args = parser.parse_args()

    # Create configuration (topic names are defined in default_config)
    config = {}

    # Execute pipeline
    pipeline = RgbDepthExtractorPipelineV2(
        bag_path=args.bag_path,
        output_dir=args.output_dir,
        config=config
    )

    try:
        pipeline.run()
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
