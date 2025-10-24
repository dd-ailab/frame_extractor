"""
GPS Data Processing Module (GPS 데이터 처리 모듈)

Handles GPS NavSatFix data extraction, validation, and quality assessment
"""

from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2

from bag_reader import RosBagReader, NavSatFixMessage


class GPSProcessor:
    """GPS data processor with quality validation and Hz calculation"""

    def __init__(self, logger, config: Dict[str, Any]):
        """
        Initialize GPS processor

        Args:
            logger: Logger instance
            config: Configuration dictionary
        """
        self.logger = logger
        self.config = config

    @staticmethod
    def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two GPS points using Haversine formula

        Args:
            lat1, lon1: First point coordinates in degrees
            lat2, lon2: Second point coordinates in degrees

        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    @staticmethod
    def validate_gps_intervals(gps_list: List[NavSatFixMessage], max_gap_ms: float = 300.0) -> List[int]:
        """
        Detect GPS message gaps

        Args:
            gps_list: List of NavSatFix messages sorted by timestamp
            max_gap_ms: Maximum acceptable gap in milliseconds

        Returns:
            List of indices where gaps exceed threshold
        """
        gap_indices = []

        for i in range(1, len(gps_list)):
            time_diff_ns = gps_list[i].timestamp - gps_list[i-1].timestamp
            time_diff_ms = time_diff_ns / 1e6

            if time_diff_ms > max_gap_ms:
                gap_indices.append(i)

        return gap_indices

    @staticmethod
    def validate_gps_hz_windowed(gps_list: List[NavSatFixMessage], window_size: int = 10, min_hz: float = 5.0, max_hz: float = 15.0) -> List[Tuple[int, float]]:
        """
        Windowed Hz analysis to detect frequency anomalies

        Args:
            gps_list: List of NavSatFix messages sorted by timestamp
            window_size: Number of messages in sliding window
            min_hz: Minimum acceptable Hz
            max_hz: Maximum acceptable Hz

        Returns:
            List of (index, hz) tuples where Hz is outside acceptable range
        """
        anomalies = []

        if len(gps_list) < window_size:
            return anomalies

        for i in range(len(gps_list) - window_size + 1):
            window = gps_list[i:i + window_size]

            time_span_ns = window[-1].timestamp - window[0].timestamp
            time_span_s = time_span_ns / 1e9

            if time_span_s > 0:
                hz = (window_size - 1) / time_span_s

                if hz < min_hz or hz > max_hz:
                    anomalies.append((i, hz))

        return anomalies

    def extract_gps_data(self, reader: RosBagReader) -> Dict[int, Dict[str, Any]]:
        """
        Extract GPS data from NavSatFix topic with quality validation

        Returns:
            Dictionary mapping timestamp to {'msg': NavSatFixMessage, 'hz': float}
        """
        gps_data = {}
        try:
            gps_topic = self.config.get('gps_topic', '/gps/fix')

            self.logger.info("  Extracting GPS NavSatFix data...")

            gps_list = []
            prev_gps_timestamp = None

            for gps_msg in tqdm(reader.read_navsatfix(gps_topic), desc="  Reading GPS"):
                # Calculate GPS Hz
                if prev_gps_timestamp is not None:
                    time_diff_sec = (gps_msg.timestamp - prev_gps_timestamp) / 1e9
                    gps_hz = 1.0 / time_diff_sec if time_diff_sec > 0 else 0.0
                else:
                    gps_hz = 0.0

                gps_list.append(gps_msg)
                gps_data[gps_msg.timestamp] = {'msg': gps_msg, 'hz': gps_hz}
                prev_gps_timestamp = gps_msg.timestamp

            self.logger.info(f"    GPS records: {len(gps_data)}")

            if len(gps_list) > 0:
                gps_list_sorted = sorted(gps_list, key=lambda x: x.timestamp)

                duration_ns = gps_list_sorted[-1].timestamp - gps_list_sorted[0].timestamp
                duration_s = duration_ns / 1e9
                avg_hz = (len(gps_list) - 1) / duration_s if duration_s > 0 else 0

                self.logger.info(f"    GPS duration: {duration_s:.2f}s, Average Hz: {avg_hz:.2f}")

                gap_indices = self.validate_gps_intervals(gps_list_sorted, max_gap_ms=300.0)
                if gap_indices:
                    self.logger.warning(f"    GPS message gaps detected at {len(gap_indices)} locations")

                hz_anomalies = self.validate_gps_hz_windowed(gps_list_sorted, window_size=10, min_hz=5.0, max_hz=15.0)
                if hz_anomalies:
                    self.logger.warning(f"    GPS Hz anomalies detected: {len(hz_anomalies)} windows")

                position_jumps = 0
                max_speed_threshold_mps = 50.0
                for i in range(1, len(gps_list_sorted)):
                    prev = gps_list_sorted[i-1]
                    curr = gps_list_sorted[i]

                    distance_m = self.calculate_haversine_distance(
                        prev.latitude, prev.longitude,
                        curr.latitude, curr.longitude
                    )

                    time_diff_s = (curr.timestamp - prev.timestamp) / 1e9
                    if time_diff_s > 0:
                        speed_mps = distance_m / time_diff_s
                        if speed_mps > max_speed_threshold_mps:
                            position_jumps += 1

                if position_jumps > 0:
                    self.logger.warning(f"    GPS position jumps detected: {position_jumps} occurrences")

        except Exception as e:
            self.logger.warning(f"  Failed to extract GPS data: {e}")

        return gps_data

    @staticmethod
    def find_closest_gps_with_quality(
        gps_data: Dict[int, Dict[str, Any]],
        target_timestamp: int,
        max_time_diff_ms: float = 200.0
    ) -> Optional[Tuple[NavSatFixMessage, float, str, float]]:
        """
        Find closest GPS data with quality assessment

        Args:
            gps_data: Dictionary of GPS data by timestamp {'msg': NavSatFixMessage, 'hz': float}
            target_timestamp: Target timestamp (nanoseconds)
            max_time_diff_ms: Maximum acceptable time difference in milliseconds

        Returns:
            Tuple of (NavSatFixMessage, time_diff_ms, quality_grade, gps_hz) or None
            Quality grades: 'EXCELLENT' (<50ms), 'GOOD' (<100ms), 'ACCEPTABLE' (<200ms), 'POOR' (>200ms)
        """
        if not gps_data:
            return None

        closest_ts = min(gps_data.keys(), key=lambda ts: abs(ts - target_timestamp))
        time_diff_ns = abs(closest_ts - target_timestamp)
        time_diff_ms = time_diff_ns / 1e6

        if time_diff_ms < 50.0:
            quality = 'EXCELLENT'
        elif time_diff_ms < 100.0:
            quality = 'GOOD'
        elif time_diff_ms < max_time_diff_ms:
            quality = 'ACCEPTABLE'
        else:
            quality = 'POOR'

        gps_entry = gps_data[closest_ts]
        gps_msg = gps_entry['msg']
        gps_hz = gps_entry['hz']

        return (gps_msg, time_diff_ms, quality, gps_hz)
