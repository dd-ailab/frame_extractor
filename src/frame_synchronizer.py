"""
Frame Synchronizer Module
RGB와 Depth 이미지를 timestamp 기반으로 동기화하는 모듈
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from bisect import bisect_left


@dataclass
class SynchronizedFrame:
    """동기화된 RGB-Depth 프레임 쌍"""
    rgb_timestamp: int  # nanoseconds
    depth_timestamp: int  # nanoseconds
    rgb_data: np.ndarray
    depth_data: np.ndarray
    time_diff: int  # nanoseconds (RGB와 Depth 사이의 timestamp 차이)


class FrameSynchronizer:
    """
    RGB와 Depth 이미지를 timestamp 기반으로 동기화하는 클래스

    동기화 방법:
    - Nearest neighbor matching
    - Time tolerance 내에서 가장 가까운 timestamp 매칭
    """

    def __init__(self, time_tolerance_ns: int = 33_000_000):  # 33ms default (30fps)
        """
        초기화

        Args:
            time_tolerance_ns: 허용 가능한 최대 timestamp 차이 (nanoseconds)
                              기본값 33ms는 30fps 기준 1프레임 간격
        """
        self.time_tolerance_ns = time_tolerance_ns

    def synchronize_frames(
        self,
        rgb_timestamps: List[int],
        rgb_frames: List[np.ndarray],
        depth_timestamps: List[int],
        depth_frames: List[np.ndarray]
    ) -> List[SynchronizedFrame]:
        """
        RGB와 Depth 프레임을 동기화

        Args:
            rgb_timestamps: RGB timestamp 리스트
            rgb_frames: RGB frame 데이터 리스트
            depth_timestamps: Depth timestamp 리스트
            depth_frames: Depth frame 데이터 리스트

        Returns:
            동기화된 프레임 쌍 리스트
        """
        if len(rgb_timestamps) == 0 or len(depth_timestamps) == 0:
            return []

        synchronized_frames = []

        # Depth timestamp를 기준으로 RGB를 매칭
        # (일반적으로 depth가 더 적은 프레임을 가지므로)
        for depth_idx, depth_ts in enumerate(depth_timestamps):
            # 가장 가까운 RGB timestamp 찾기
            rgb_idx, time_diff = self._find_nearest_timestamp(
                depth_ts,
                rgb_timestamps
            )

            # Time tolerance 체크
            if abs(time_diff) <= self.time_tolerance_ns:
                synchronized_frames.append(SynchronizedFrame(
                    rgb_timestamp=rgb_timestamps[rgb_idx],
                    depth_timestamp=depth_ts,
                    rgb_data=rgb_frames[rgb_idx],
                    depth_data=depth_frames[depth_idx],
                    time_diff=time_diff
                ))

        return synchronized_frames

    def synchronize_frames_fast(
        self,
        rgb_timestamps: np.ndarray,
        rgb_frames: Optional[List[np.ndarray]],
        depth_timestamps: np.ndarray,
        depth_frames: Optional[List[np.ndarray]]
    ) -> List[SynchronizedFrame]:
        """
        Sliding Window 알고리즘을 사용한 고속 동기화 

        Args:
            rgb_timestamps: RGB timestamp 배열 (정렬된 상태)
            rgb_frames: RGB frame 리스트 (None이면 lazy loading)
            depth_timestamps: Depth timestamp 배열 (정렬된 상태)
            depth_frames: Depth frame 리스트 (None이면 lazy loading)

        Returns:
            동기화된 프레임 쌍 리스트
        """
        if len(rgb_timestamps) == 0 or len(depth_timestamps) == 0:
            return []

        synchronized_frames = []
        rgb_idx = 0
        n_rgb = len(rgb_timestamps)
        n_depth = len(depth_timestamps)

        # Two-pointer sliding window
        for depth_idx in range(n_depth):
            depth_ts = depth_timestamps[depth_idx]

            # RGB pointer를 depth에 가장 가까운 위치로 이동
            while rgb_idx < n_rgb - 1:
                current_diff = abs(rgb_timestamps[rgb_idx] - depth_ts)
                next_diff = abs(rgb_timestamps[rgb_idx + 1] - depth_ts)

                if next_diff < current_diff:
                    rgb_idx += 1
                else:
                    break

            # 최적 매칭 확인
            time_diff = rgb_timestamps[rgb_idx] - depth_ts

            if abs(time_diff) <= self.time_tolerance_ns:
                # Lazy loading 지원
                rgb_data = rgb_frames[rgb_idx] if rgb_frames is not None else None
                depth_data = depth_frames[depth_idx] if depth_frames is not None else None

                synchronized_frames.append(SynchronizedFrame(
                    rgb_timestamp=int(rgb_timestamps[rgb_idx]),
                    depth_timestamp=int(depth_ts),
                    rgb_data=rgb_data,
                    depth_data=depth_data,
                    time_diff=int(time_diff)
                ))

        return synchronized_frames

    def _find_nearest_timestamp(
        self,
        target_timestamp: int,
        timestamp_list: List[int]
    ) -> Tuple[int, int]:
        """
        가장 가까운 timestamp의 index 찾기 
        Args:
            target_timestamp: 찾고자 하는 timestamp
            timestamp_list: timestamp 리스트 (정렬된 상태)

        Returns:
            (index, time_difference)
        """
        # Binary search로 insertion point 찾기
        pos = bisect_left(timestamp_list, target_timestamp)

        # Edge cases 처리
        if pos == 0:
            return 0, timestamp_list[0] - target_timestamp
        if pos == len(timestamp_list):
            return len(timestamp_list) - 1, timestamp_list[-1] - target_timestamp

        # 이전 index와 현재 index 중 더 가까운 것 선택
        before = pos - 1
        after = pos

        diff_before = abs(timestamp_list[before] - target_timestamp)
        diff_after = abs(timestamp_list[after] - target_timestamp)

        if diff_before < diff_after:
            return before, timestamp_list[before] - target_timestamp
        else:
            return after, timestamp_list[after] - target_timestamp

    def filter_by_time_range(
        self,
        frames: List[SynchronizedFrame],
        start_time: int,
        end_time: int
    ) -> List[SynchronizedFrame]:
        """
        특정 시간 범위 내의 프레임만 필터링

        Args:
            frames: 동기화된 프레임 리스트
            start_time: 시작 시간 (nanoseconds)
            end_time: 종료 시간 (nanoseconds)

        Returns:
            필터링된 프레임 리스트
        """
        filtered = []
        for frame in frames:
            # RGB timestamp 기준으로 필터링
            if start_time <= frame.rgb_timestamp <= end_time:
                filtered.append(frame)

        return filtered

    def get_synchronization_statistics(
        self,
        frames: List[SynchronizedFrame]
    ) -> dict:
        """
        동기화 통계 정보 계산

        Args:
            frames: 동기화된 프레임 리스트

        Returns:
            통계 정보 dictionary
        """
        if len(frames) == 0:
            return {
                'total_pairs': 0,
                'mean_time_diff_ms': 0.0,
                'max_time_diff_ms': 0.0,
                'std_time_diff_ms': 0.0
            }

        time_diffs = np.array([abs(f.time_diff) for f in frames])

        # nanoseconds를 milliseconds로 변환
        time_diffs_ms = time_diffs / 1e6

        return {
            'total_pairs': len(frames),
            'mean_time_diff_ms': float(np.mean(time_diffs_ms)),
            'max_time_diff_ms': float(np.max(time_diffs_ms)),
            'std_time_diff_ms': float(np.std(time_diffs_ms)),
            'median_time_diff_ms': float(np.median(time_diffs_ms))
        }
