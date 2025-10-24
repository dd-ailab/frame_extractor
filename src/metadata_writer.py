"""
Metadata Writing Module (메타데이터 저장 모듈)

Handles saving frames, metadata, camera calibration, and extraction summaries
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

from frame_synchronizer import SynchronizedFrame
from bag_reader import NavSatFixMessage
from gps_processor import GPSProcessor
from utils import find_closest_dict, extract_dict_values


class MetadataWriter:
    """Writer for frames, metadata, and summaries"""

    # Frame ID format constant
    FRAME_ID_FORMAT = "frame_{:06d}"

    # Time conversion constants
    NS_TO_MS = 1e6  # nanoseconds → milliseconds

    def __init__(self, logger, config: Dict[str, Any], extractor, output_dir: str, all_metadata_records: List, camera_calibration: Dict[str, Any], bag_path: str):
        """
        Initialize metadata writer

        Args:
            logger: Logger instance
            config: Configuration dictionary
            extractor: FrameExtractor instance for saving images
            output_dir: Output directory path
            all_metadata_records: Reference to metadata records list
            camera_calibration: Reference to camera calibration dictionary
            bag_path: ROS bag file path
        """
        self.logger = logger
        self.config = config
        self.extractor = extractor
        self.output_dir = output_dir
        self.all_metadata_records = all_metadata_records
        self.camera_calibration = camera_calibration
        self.bag_path = bag_path

    def save_all_frames(
        self,
        all_frames: List[SynchronizedFrame],
        imu_data: Dict[int, Dict[str, float]],
        rs_metadata: Dict[int, Dict[str, Any]],
        gps_data: Dict[int, Dict[str, Any]]
    ) -> int:
        """
        향상된 metadata와 함께 모든 frame 저장
        Save all frames with enhanced metadata

        각 frame에 대해:
        - RGB/Depth 이미지 저장
        - GPS, IMU 가속도, RealSense metadata 통합
        """
        total_extracted = 0

        self.logger.info(f"  Processing {len(all_frames)} frames...")

        # DEBUG: RGB 프레임 타임스탬프 범위 로깅
        if all_frames:
            rgb_timestamps = [f.rgb_timestamp for f in all_frames]
            min_rgb_ts = min(rgb_timestamps)
            max_rgb_ts = max(rgb_timestamps)
            self.logger.info(f"  DEBUG: RGB timestamp range: {min_rgb_ts} to {max_rgb_ts}")
            self.logger.info(f"  DEBUG: Total frames to process: {len(all_frames)}")

        # DEBUG: IMU 데이터 상태 확인
        self.logger.info(f"  DEBUG: IMU data size: {len(imu_data)} records")
        self.logger.info(f"  DEBUG: GPS data size: {len(gps_data)} records")

        # 이전 프레임 timestamp 추적 (Track previous frame timestamps for Hz calculation)
        prev_rgb_timestamp = None
        prev_depth_timestamp = None

        # 모든 frame을 순차적으로 처리 (Process all frames sequentially)
        for frame_idx, frame in enumerate(tqdm(all_frames, desc="  Saving frames")):
            # Frame ID 생성 (Generate frame ID - new format: frame_000001)
            frame_id = self.FRAME_ID_FORMAT.format(frame_idx + 1)

            # RGB 및 Depth 이미지 저장 (Save RGB and Depth images)
            rgb_path = self.extractor.rgb_dir / f"{frame_id}.{self.config['rgb_format']}"
            depth_path = self.extractor.depth_dir / f"{frame_id}.{self.config['depth_format']}"

            self.extractor._save_rgb_image(frame.rgb_data, rgb_path)
            if self.config['depth_format'] == 'png':
                self.extractor._save_depth_image_png(frame.depth_data, depth_path)
            else:
                self.extractor._save_depth_image_npy(frame.depth_data, depth_path)

            # Depth visualization 저장 (Save depth visualization)
            if self.config['save_visualization']:
                vis_path = self.extractor.vis_dir / f"{frame_id}_vis.png"
                self.extractor._save_depth_visualization(frame.depth_data, vis_path)

            # Calculate RGB and Depth Hz
            if prev_rgb_timestamp is not None:
                time_diff_sec = (frame.rgb_timestamp - prev_rgb_timestamp) / 1e9
                rgb_hz = 1.0 / time_diff_sec if time_diff_sec > 0 else 0.0
            else:
                rgb_hz = 0.0

            if prev_depth_timestamp is not None:
                time_diff_sec = (frame.depth_timestamp - prev_depth_timestamp) / 1e9
                depth_hz = 1.0 / time_diff_sec if time_diff_sec > 0 else 0.0
            else:
                depth_hz = 0.0

            prev_rgb_timestamp = frame.rgb_timestamp
            prev_depth_timestamp = frame.depth_timestamp

            # 가장 가까운 GPS, IMU metadata 찾기 (Find closest GPS, IMU metadata)
            gps_result = GPSProcessor.find_closest_gps_with_quality(gps_data, frame.rgb_timestamp)
            closest_imu = find_closest_dict(imu_data, frame.rgb_timestamp)

            # GPS data extraction with quality and Hz
            if gps_result:
                gps_msg, gps_time_diff_ms, gps_quality, gps_hz = gps_result
                gps_lat = gps_msg.latitude
                gps_lon = gps_msg.longitude
                gps_alt = gps_msg.altitude
                gps_status = gps_msg.status
            else:
                gps_lat = 0.0
                gps_lon = 0.0
                gps_alt = 0.0
                gps_status = -1
                gps_time_diff_ms = 0.0
                gps_quality = 'NONE'
                gps_hz = 0.0

            # DEBUG: 첫 3개 프레임의 매칭 결과 로깅
            if frame_idx < 3:
                self.logger.info(f"  DEBUG: Frame {frame_idx} - RGB timestamp: {frame.rgb_timestamp}")
                self.logger.info(f"  DEBUG: Frame {frame_idx} - GPS: lat={gps_lat}, lon={gps_lon}, alt={gps_alt}, quality={gps_quality}")
                self.logger.info(f"  DEBUG: Frame {frame_idx} - closest_imu: {closest_imu}")

            # IMU acceleration 추출
            imu_values = extract_dict_values(closest_imu, {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'magnitude': 0.0
            })

            # Extract IMU Hz
            imu_hz = closest_imu.get('hz', 0.0) if closest_imu else 0.0

            # DEBUG: 첫 3개 프레임의 추출된 IMU 값 로깅
            if frame_idx < 3:
                self.logger.info(f"  DEBUG: Frame {frame_idx} - imu_values: {imu_values}")
                self.logger.info(f"  DEBUG: Frame {frame_idx} - imu_hz: {imu_hz}")

            # Metadata 레코드 생성 (Build metadata record - RGB-D, IMU, GPS, timestamp only)
            metadata_record = {
                'frame_id': frame_id,
                'frame_index': frame_idx + 1,
                'rgb_timestamp_ns': frame.rgb_timestamp,
                'depth_timestamp_ns': frame.depth_timestamp,
                'time_diff_ms': frame.time_diff / self.NS_TO_MS,
                'rgb_path': f"rgb/{frame_id}.{self.config['rgb_format']}",
                'depth_path': f"depth/{frame_id}.{self.config['depth_format']}",
                'rgb_hz': rgb_hz,
                'depth_hz': depth_hz,
                'gps_latitude': gps_lat,
                'gps_longitude': gps_lon,
                'gps_altitude': gps_alt,
                'gps_status': gps_status,
                'gps_time_diff_ms': gps_time_diff_ms,
                'gps_match_quality': gps_quality,
                'gps_hz': gps_hz,
                'imu_accel_x': imu_values['x'],
                'imu_accel_y': imu_values['y'],
                'imu_accel_z': imu_values['z'],
                'imu_accel_mag': imu_values['magnitude'],
                'imu_hz': imu_hz
            }

            self.all_metadata_records.append(metadata_record)
            total_extracted += 1

        return total_extracted

    def save_unified_metadata(self) -> None:
        """Save unified metadata_all.csv"""
        if not self.all_metadata_records:
            self.logger.warning("No metadata records to save")
            return

        metadata_df = pd.DataFrame(self.all_metadata_records)
        metadata_path = Path(self.output_dir) / 'metadata_all.csv'
        metadata_df.to_csv(metadata_path, index=False)
        self.logger.info(f"  Saved unified metadata: {metadata_path}")

    def save_camera_calibration(self) -> None:
        """Save camera_calibration.json"""
        if not self.camera_calibration:
            return

        calib_path = Path(self.output_dir) / 'camera_calibration.json'
        with open(calib_path, 'w') as f:
            json.dump(self.camera_calibration, f, indent=2)
        self.logger.info(f"  Saved camera calibration: {calib_path}")

    def save_extraction_summary(self, sync_stats: Dict[str, Any], total_frames: int) -> None:
        """Save extraction_summary.json"""
        summary = {
            'bag_path': str(self.bag_path),
            'total_extracted_frames': total_frames,
            'config': self.config,
            'synchronization_stats': sync_stats,
            'camera_calibration_available': len(self.camera_calibration) > 0,
            'gps_data_available': self.config['save_gps_metadata']
        }

        summary_path = Path(self.output_dir) / 'extraction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"  Saved extraction summary: {summary_path}")
