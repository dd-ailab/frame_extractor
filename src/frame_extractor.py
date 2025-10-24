"""
Frame Extractor Module
Save synchronized RGB-Depth frames to filesystem
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from frame_synchronizer import SynchronizedFrame

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Class to save synchronized frames to filesystem

    Directory structure:
    output_dir/
      ├── rgb/
      │   ├── 000000.png
      │   ├── 000001.png
      │   └── ...
      ├── depth/
      │   ├── 000000.png  (16-bit)
      │   └── ...
      └── metadata.csv
    """

    def __init__(
        self,
        output_dir: str,
        rgb_format: str = 'png',
        depth_format: str = 'png',
        save_visualization: bool = False
    ):
        """
        Initialize

        Args:
            output_dir: Output directory path
            rgb_format: RGB image save format ('png' or 'jpg')
            depth_format: Depth image save format ('png' or 'npy')
            save_visualization: Whether to save depth visualization images
        """
        self.output_dir = Path(output_dir)
        self.rgb_format = rgb_format
        self.depth_format = depth_format
        self.save_visualization = save_visualization
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create directories
        self.rgb_dir = self.output_dir / 'rgb'
        self.depth_dir = self.output_dir / 'depth'
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        if save_visualization:
            self.vis_dir = self.output_dir / 'depth_vis'
            self.vis_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(
        self,
        frames: List[SynchronizedFrame],
        segment_id: int = 0,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Save frames to files

        Args:
            frames: List of synchronized frames to save
            segment_id: Segment identifier
            description: Segment description

        Returns:
            Extraction statistics
        """
        if len(frames) == 0:
            return {'total_frames': 0}

        metadata_records = []

        self.logger.info(f"Extracting frames... (total: {len(frames)})")
        for idx, frame in enumerate(tqdm(frames, desc="Extracting frames")):
            # 파일명 생성 (segment_id와 frame index 포함)
            frame_id = f"{segment_id:03d}_{idx:06d}"

            # RGB 이미지 저장
            rgb_path = self.rgb_dir / f"{frame_id}.{self.rgb_format}"
            self._save_rgb_image(frame.rgb_data, rgb_path)

            # Depth 이미지 저장
            if self.depth_format == 'png':
                depth_path = self.depth_dir / f"{frame_id}.png"
                self._save_depth_image_png(frame.depth_data, depth_path)
            elif self.depth_format == 'npy':
                depth_path = self.depth_dir / f"{frame_id}.npy"
                self._save_depth_image_npy(frame.depth_data, depth_path)

            # Depth visualization 저장 (optional)
            if self.save_visualization:
                vis_path = self.vis_dir / f"{frame_id}_vis.png"
                self._save_depth_visualization(frame.depth_data, vis_path)

            # Metadata 기록
            metadata_records.append({
                'frame_id': frame_id,
                'segment_id': segment_id,
                'index': idx,
                'rgb_timestamp_ns': frame.rgb_timestamp,
                'depth_timestamp_ns': frame.depth_timestamp,
                'time_diff_ns': frame.time_diff,
                'time_diff_ms': frame.time_diff / 1e6,
                'rgb_path': str(rgb_path.relative_to(self.output_dir)),
                'depth_path': str(depth_path.relative_to(self.output_dir))
            })

        # Metadata CSV 저장
        metadata_df = pd.DataFrame(metadata_records)
        metadata_csv_path = self.output_dir / f'metadata_segment_{segment_id:03d}.csv'
        metadata_df.to_csv(metadata_csv_path, index=False)

        # 통합 metadata 업데이트
        self._update_combined_metadata(metadata_df)

        # 통계 정보
        stats = {
            'total_frames': len(frames),
            'segment_id': segment_id,
            'description': description,
            'output_dir': str(self.output_dir),
            'metadata_path': str(metadata_csv_path)
        }

        return stats

    def _save_rgb_image(self, rgb_data: np.ndarray, path: Path):
        """RGB 이미지 저장"""
        # BGR로 변환 (OpenCV는 BGR 사용)
        if rgb_data.shape[2] == 3:
            bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        else:
            bgr_data = rgb_data

        if self.rgb_format == 'jpg':
            cv2.imwrite(str(path), bgr_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(str(path), bgr_data)

    def _save_depth_image_png(self, depth_data: np.ndarray, path: Path):
        """
        Depth 이미지를 16-bit PNG로 저장

        RealSense depth는 일반적으로 millimeter 단위
        """
        # uint16으로 변환
        if depth_data.dtype != np.uint16:
            depth_data = depth_data.astype(np.uint16)

        cv2.imwrite(str(path), depth_data)

    def _save_depth_image_npy(self, depth_data: np.ndarray, path: Path):
        """Depth 이미지를 NumPy 파일로 저장"""
        np.save(str(path), depth_data)

    def _save_depth_visualization(self, depth_data: np.ndarray, path: Path):
        """
        Depth 이미지의 시각화 버전 저장 (colormap 적용)

        Args:
            depth_data: depth 데이터
            path: 저장 경로
        """
        # Normalize to 0-255
        depth_normalized = cv2.normalize(
            depth_data,
            None,
            0, 255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        cv2.imwrite(str(path), depth_colored)

    def _update_combined_metadata(self, new_metadata: pd.DataFrame):
        """
        통합 metadata 파일 업데이트

        Args:
            new_metadata: 새로운 metadata DataFrame
        """
        combined_path = self.output_dir / 'metadata_all.csv'

        if combined_path.exists():
            # 기존 metadata 읽기
            existing_df = pd.read_csv(combined_path)
            # 새로운 데이터 추가
            combined_df = pd.concat([existing_df, new_metadata], ignore_index=True)
        else:
            combined_df = new_metadata

        # 저장
        combined_df.to_csv(combined_path, index=False)

    def save_extraction_summary(
        self,
        summary_data: Dict[str, Any],
        filename: str = 'extraction_summary.json'
    ):
        """
        Save overall extraction summary

        Args:
            summary_data: Summary information dictionary
            filename: Filename to save
        """
        summary_path = self.output_dir / filename

        # Add timestamp
        summary_data['extraction_time'] = datetime.now().isoformat()

        # Convert NumPy types to Python native types for JSON serialization
        summary_data = self._convert_numpy_types(summary_data)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Extraction summary saved: {summary_path}")

    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        Recursively convert NumPy types to Python native types for JSON serialization

        Args:
            obj: Object to convert (dict, list, numpy type, or primitive)

        Returns:
            Converted object with Python native types
        """
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def extract_frames_parallel(
        self,
        frames: List[SynchronizedFrame],
        segment_id: int = 0,
        description: str = "",
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        High-speed frame saving with parallel processing

        Args:
            frames: List of synchronized frames to save
            segment_id: Segment identifier
            description: Segment description
            max_workers: Number of parallel workers (default: 4)

        Returns:
            Extraction statistics
        """
        if len(frames) == 0:
            return {'total_frames': 0}

        self.logger.info(f"Parallel frame extraction... (total: {len(frames)}, workers: {max_workers})")

        # 저장 작업 준비
        save_tasks = []
        for idx, frame in enumerate(frames):
            frame_id = f"{segment_id:03d}_{idx:06d}"

            rgb_path = self.rgb_dir / f"{frame_id}.{self.rgb_format}"
            depth_path = self.depth_dir / f"{frame_id}.{self.depth_format if self.depth_format == 'png' else 'png'}"

            task = {
                'frame_id': frame_id,
                'rgb_data': frame.rgb_data,
                'depth_data': frame.depth_data,
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'rgb_timestamp': frame.rgb_timestamp,
                'depth_timestamp': frame.depth_timestamp,
                'time_diff': frame.time_diff,
                'rgb_format': self.rgb_format,
                'depth_format': self.depth_format,
                'save_visualization': self.save_visualization,
                'vis_dir': self.vis_dir if self.save_visualization else None
            }
            save_tasks.append(task)

        # 병렬 저장 실행
        metadata_records = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_save_frame_worker, task) for task in save_tasks]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Saving frames"):
                metadata = future.result()
                metadata_records.append(metadata)

        # Segment ID 순서로 정렬
        metadata_records.sort(key=lambda x: x['frame_id'])

        # Metadata CSV 저장
        metadata_df = pd.DataFrame(metadata_records)

        # 상대 경로로 변환
        metadata_df['rgb_path'] = metadata_df['rgb_path'].apply(
            lambda p: str(Path(p).relative_to(self.output_dir))
        )
        metadata_df['depth_path'] = metadata_df['depth_path'].apply(
            lambda p: str(Path(p).relative_to(self.output_dir))
        )

        metadata_csv_path = self.output_dir / f'metadata_segment_{segment_id:03d}.csv'
        metadata_df.to_csv(metadata_csv_path, index=False)

        # 통합 metadata 업데이트
        self._update_combined_metadata(metadata_df)

        # 통계 정보
        stats = {
            'total_frames': len(frames),
            'segment_id': segment_id,
            'description': description,
            'output_dir': str(self.output_dir),
            'metadata_path': str(metadata_csv_path)
        }

        return stats


def _save_frame_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Frame save worker function for parallel processing

    Args:
        task: Save task information

    Returns:
        metadata dictionary
    """
    # RGB 저장
    rgb_data = task['rgb_data']
    if rgb_data.shape[2] == 3 if len(rgb_data.shape) == 3 else False:
        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
    else:
        bgr_data = rgb_data

    if task['rgb_format'] == 'jpg':
        cv2.imwrite(str(task['rgb_path']), bgr_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(task['rgb_path']), bgr_data)

    # Depth 저장
    depth_data = task['depth_data']
    if task['depth_format'] == 'png':
        if depth_data.dtype != np.uint16:
            depth_data = depth_data.astype(np.uint16)
        cv2.imwrite(str(task['depth_path']), depth_data)
    elif task['depth_format'] == 'npy':
        np.save(str(task['depth_path']), depth_data)

    # Visualization 저장 (optional)
    if task['save_visualization']:
        vis_path = task['vis_dir'] / f"{task['frame_id']}_vis.png"
        depth_normalized = cv2.normalize(
            depth_data,
            None,
            0, 255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(str(vis_path), depth_colored)

    # Metadata 반환
    return {
        'frame_id': task['frame_id'],
        'segment_id': int(task['frame_id'].split('_')[0]),
        'index': int(task['frame_id'].split('_')[1]),
        'rgb_timestamp_ns': task['rgb_timestamp'],
        'depth_timestamp_ns': task['depth_timestamp'],
        'time_diff_ns': task['time_diff'],
        'time_diff_ms': task['time_diff'] / 1e6,
        'rgb_path': str(task['rgb_path']),
        'depth_path': str(task['depth_path'])
    }


def load_depth_image(path: Path) -> np.ndarray:
    """
    Load saved depth image

    Args:
        path: Depth image path (.png or .npy)

    Returns:
        Depth data (numpy array)
    """
    if path.suffix == '.npy':
        return np.load(str(path))
    elif path.suffix == '.png':
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError(f"Unsupported depth format: {path.suffix}")


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load metadata CSV file

    Args:
        metadata_path: Metadata CSV file path

    Returns:
        Metadata DataFrame
    """
    return pd.read_csv(metadata_path)
