"""
Sensor Data Extraction Module (센서 데이터 추출 모듈)

Handles extraction of IMU acceleration data, RealSense metadata, and camera calibration
"""

from typing import Dict, Any
import numpy as np
from tqdm import tqdm

from bag_reader import RosBagReader


class SensorExtractor:
    """Extract sensor data from ROS bag (IMU, RealSense metadata, Camera info)"""

    def __init__(self, logger, config: Dict[str, Any], camera_calibration: Dict[str, Any]):
        """
        Initialize sensor extractor

        Args:
            logger: Logger instance
            config: Configuration dictionary
            camera_calibration: Reference to camera calibration dictionary
        """
        self.logger = logger
        self.config = config
        self.camera_calibration = camera_calibration

    def extract_imu_acceleration(self, reader: RosBagReader) -> Dict[int, Dict[str, float]]:
        """
        Extract IMU acceleration data (for metadata, not for filtering)

        Returns:
            IMU acceleration dictionary: {timestamp: {'x': val, 'y': val, 'z': val, 'magnitude': val, 'hz': val}}
        """
        self.logger.info("[2/6] Extracting IMU acceleration data...")
        imu_data = {}
        prev_imu_timestamp = None

        for imu_msg in tqdm(reader.read_imu(self.config['imu_topic']), desc="  Reading IMU"):
            acc_x = float(imu_msg.linear_acceleration[0])
            acc_y = float(imu_msg.linear_acceleration[1])
            acc_z = float(imu_msg.linear_acceleration[2])
            acc_mag = float(np.linalg.norm(imu_msg.linear_acceleration))

            # Calculate IMU Hz
            if prev_imu_timestamp is not None:
                time_diff_sec = (imu_msg.timestamp - prev_imu_timestamp) / 1e9
                imu_hz = 1.0 / time_diff_sec if time_diff_sec > 0 else 0.0
            else:
                imu_hz = 0.0

            imu_data[imu_msg.timestamp] = {
                'x': acc_x,
                'y': acc_y,
                'z': acc_z,
                'magnitude': acc_mag,
                'hz': imu_hz
            }

            prev_imu_timestamp = imu_msg.timestamp

        self.logger.info(f"  Extracted {len(imu_data)} IMU acceleration records")

        # DEBUG: IMU timestamp range logging
        if imu_data:
            min_ts = min(imu_data.keys())
            max_ts = max(imu_data.keys())
            self.logger.info(f"  DEBUG: IMU timestamp range: {min_ts} to {max_ts}")
            # Log first 3 IMU data samples
            sample_items = list(imu_data.items())[:3]
            for ts, values in sample_items:
                self.logger.info(f"  DEBUG: IMU sample - ts={ts}, x={values['x']:.3f}, y={values['y']:.3f}, z={values['z']:.3f}")
        else:
            self.logger.warning("  DEBUG: IMU data dictionary is EMPTY!")

        return imu_data

    def extract_realsense_metadata(self, reader: RosBagReader) -> Dict[int, Dict[str, Any]]:
        """
        Extract RealSense camera metadata

        Returns:
            RealSense metadata dictionary: {timestamp: {'frame_counter': val, 'sensor_timestamp': val, ...}}
        """
        self.logger.info("[3/6] Extracting RealSense metadata...")
        rs_metadata = {}

        try:
            for metadata_msg in tqdm(reader.read_metadata(self.config['metadata_topic']), desc="  Reading Metadata"):
                rs_metadata[metadata_msg.timestamp] = {
                    'frame_counter': metadata_msg.frame_counter,
                    'frame_timestamp': metadata_msg.frame_timestamp,
                    'exposure_time': metadata_msg.exposure_time,
                    'gain': metadata_msg.gain,
                    'brightness': metadata_msg.brightness
                }

            self.logger.info(f"  Extracted {len(rs_metadata)} RealSense metadata records")
        except Exception as e:
            self.logger.warning(f"  Failed to extract RealSense metadata: {e}")

        return rs_metadata

    def extract_camera_info(self, reader: RosBagReader) -> None:
        """Extract camera calibration parameters"""
        try:
            self._extract_single_camera_info(
                reader,
                self.config.get('rgb_camera_info_topic'),
                'rgb'
            )
            self._extract_single_camera_info(
                reader,
                self.config.get('depth_camera_info_topic'),
                'depth'
            )

            self.logger.info(f"  Extracted camera calibration: {list(self.camera_calibration.keys())}")
        except Exception as e:
            self.logger.warning(f"  Failed to extract camera info: {e}")

    def _extract_single_camera_info(
        self,
        reader: RosBagReader,
        topic: str,
        camera_name: str
    ) -> None:
        """Extract calibration info for a single camera"""
        if not topic:
            return

        for camera_info in reader.read_camera_info(topic):
            self.camera_calibration[camera_name] = {
                'width': int(camera_info.width),
                'height': int(camera_info.height),
                'K': camera_info.K.tolist(),
                'D': camera_info.D.tolist(),
                'R': camera_info.R.tolist(),
                'P': camera_info.P.tolist()
            }
            break  # Only need first message
