#!/usr/bin/env python3
"""
Manual Frame Selection Tool

Interactive tool for manually selecting RGB-Depth frames
- Optional auto-sampling by interval/FPS before manual selection
- Display RGB and Depth images side-by-side
- Press 'y' to accept, 'n' to reject, 'q' to quit
- Save selected frames to new directory
"""

import logging
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameSampler:
    """
    Frame sampler for reducing frame count based on time interval or FPS
    (Integrated from frame_sampler.py for pre-filtering before manual selection)
    """

    @staticmethod
    def apply_metadata_sampling(
        metadata_df: pd.DataFrame,
        sample_interval: Optional[float] = None,
        sample_fps: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Apply frame sampling to metadata DataFrame

        Args:
            metadata_df: Metadata DataFrame with rgb_timestamp_ns column
            sample_interval: Time interval in seconds (takes precedence)
            sample_fps: Target FPS for sampling

        Returns:
            Sampled DataFrame
        """
        if metadata_df.empty:
            return metadata_df

        # If no sampling config, return all metadata
        if sample_interval is None and sample_fps is None:
            return metadata_df

        sampled_indices = []

        # Method 1: Time interval based (more precise, takes precedence)
        if sample_interval is not None:
            logger.info(f"Applying time-interval sampling: {sample_interval}s")

            last_saved_timestamp = None
            interval_ns = int(sample_interval * 1e9)  # Convert to nanoseconds

            for idx, row in metadata_df.iterrows():
                timestamp = row['rgb_timestamp_ns']

                if last_saved_timestamp is None:
                    # Always save first frame
                    sampled_indices.append(idx)
                    last_saved_timestamp = timestamp
                elif timestamp - last_saved_timestamp >= interval_ns:
                    # Save metadata if interval has passed
                    sampled_indices.append(idx)
                    last_saved_timestamp = timestamp

        # Method 2: FPS-based sampling (simpler, uses frame skipping)
        elif sample_fps is not None:
            logger.info(f"Applying FPS-based sampling: {sample_fps} FPS")

            # Calculate average FPS from timestamps
            if len(metadata_df) > 1:
                duration_ns = metadata_df['rgb_timestamp_ns'].iloc[-1] - metadata_df['rgb_timestamp_ns'].iloc[0]
                duration_s = duration_ns / 1e9
                current_fps = len(metadata_df) / duration_s

                # Calculate skip interval
                skip_interval = max(1, int(current_fps / sample_fps))

                logger.info(f"  Current FPS: {current_fps:.2f}, Skip interval: {skip_interval}")

                # Sample every Nth metadata
                for idx in range(0, len(metadata_df), skip_interval):
                    sampled_indices.append(metadata_df.index[idx])
            else:
                # If only 1 frame, just return it
                sampled_indices = list(metadata_df.index)

        sampled_df = metadata_df.loc[sampled_indices].reset_index(drop=True)

        logger.info(f"  Input frames: {len(metadata_df)}, Sampled frames: {len(sampled_df)} ({len(sampled_df)/len(metadata_df)*100:.1f}%)")

        return sampled_df


class ManualFrameSelector:
    """
    Interactive manual frame selection tool
    """

    def __init__(
        self,
        base_dir: Path,
        metadata_csv: Path,
        output_dir: Path,
        window_width: int = 1920,
        window_height: int = 1080
    ):
        """
        Initialize manual frame selector

        Args:
            base_dir: Base directory containing rgb/ and depth/ folders
            metadata_csv: Path to selected_frames_metadata.csv
            output_dir: Output directory for selected frames
            window_width: Display window width
            window_height: Display window height
        """
        self.base_dir = Path(base_dir)
        self.metadata_csv = Path(metadata_csv)
        self.output_dir = Path(output_dir)
        self.window_width = window_width
        self.window_height = window_height

        # Validate inputs
        self._validate_inputs()

        # Create output directories
        self.output_rgb_dir = self.output_dir / 'rgb'
        self.output_depth_dir = self.output_dir / 'depth'
        self.output_rgb_dir.mkdir(parents=True, exist_ok=True)
        self.output_depth_dir.mkdir(parents=True, exist_ok=True)

        # Selection tracking
        self.selected_frames = []
        self.rejected_frames = []
        self.current_index = 0

        # Hue adjustment tracking
        self.hue_adjustments = {}  # {frame_index: hue_delta}
        self.current_hue_delta = 0  # Current frame's hue adjustment (-180 to +180)

    def _validate_inputs(self):
        """Validate input paths"""
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Metadata CSV: {self.metadata_csv}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_metadata(self) -> pd.DataFrame:
        """Load metadata CSV"""
        logger.info(f"Loading metadata from {self.metadata_csv}")
        df = pd.read_csv(self.metadata_csv)
        logger.info(f"Loaded {len(df)} frames")
        return df

    def apply_hue_adjustment(self, rgb_image: np.ndarray, hue_delta: int) -> np.ndarray:
        """
        Apply hue adjustment to RGB image

        Args:
            rgb_image: RGB image (BGR format from cv2.imread)
            hue_delta: Hue adjustment delta value (-180 to +180)

        Returns:
            Hue-adjusted RGB image (BGR format)
        """
        # Debug: Print hue adjustment value
        logger.debug(f"apply_hue_adjustment called with hue_delta={hue_delta}")

        if hue_delta == 0:
            return rgb_image.copy()

        # Convert BGR to HSV
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Adjust hue channel
        # OpenCV HSV: H range is 0-179 (360 degrees mapped to 0-179)
        h, s, v = cv2.split(hsv)

        # Apply delta with wrapping (0-179 range)
        # Convert delta from -180~+180 to OpenCV range (-90~+90)
        hue_delta_cv = int(hue_delta / 2)
        h = h.astype(np.int32)  # Use int32 to handle negative values properly
        h = h + hue_delta_cv
        # Properly wrap around 0-179 range (handles negative values)
        h = np.mod(h, 180)
        h = h.astype(np.uint8)

        # Merge channels
        hsv_adjusted = cv2.merge([h, s, v])

        # Convert back to BGR
        rgb_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

        logger.debug(f"Hue adjustment applied successfully")
        return rgb_adjusted

    def visualize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert depth to colorized visualization

        Args:
            depth: Depth image (16-bit)

        Returns:
            Colorized depth image (BGR)
        """
        # Normalize to 0-255
        depth_normalized = cv2.normalize(
            depth,
            None,
            0, 255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        return depth_colored

    def create_display_image(
        self,
        rgb_path: Path,
        depth_path: Path,
        frame_info: str
    ) -> Tuple[np.ndarray, bool]:
        """
        Create side-by-side display of RGB and Depth

        Args:
            rgb_path: Path to RGB image
            depth_path: Path to Depth image
            frame_info: Frame information text

        Returns:
            Tuple of (display_image, success)
        """
        # Load images
        rgb = cv2.imread(str(rgb_path))
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        if rgb is None or depth is None:
            logger.error(f"Failed to load images: RGB={rgb_path}, Depth={depth_path}")
            return None, False

        # Apply hue adjustment to RGB
        rgb = self.apply_hue_adjustment(rgb, self.current_hue_delta)

        # Colorize depth
        depth_vis = self.visualize_depth(depth)

        # Resize to fit window
        target_height = self.window_height - 100  # Reserve space for text
        target_width = self.window_width // 2

        # Calculate scale to fit
        rgb_h, rgb_w = rgb.shape[:2]
        scale = min(target_width / rgb_w, target_height / rgb_h)

        new_width = int(rgb_w * scale)
        new_height = int(rgb_h * scale)

        rgb_resized = cv2.resize(rgb, (new_width, new_height))
        depth_resized = cv2.resize(depth_vis, (new_width, new_height))

        # Create side-by-side display
        display = np.hstack([rgb_resized, depth_resized])

        # Add info text
        info_panel = np.zeros((100, display.shape[1], 3), dtype=np.uint8)

        # Add instructions
        instructions = "Press: [Y] Accept | [N] Reject | [Q] Quit | [B] Back | [R] Reset Hue"
        cv2.putText(
            info_panel,
            instructions,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 100),
            2
        )

        # Add hue adjustment info
        hue_sign = "+" if self.current_hue_delta >= 0 else ""
        hue_info = f"Hue: {hue_sign}{self.current_hue_delta}"
        cv2.putText(
            info_panel,
            hue_info,
            (display.shape[1] - 200, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 100),
            2
        )

        # Add hue adjustment controls guide
        controls_text = "Arrows: +-5 | []: +-30 | Wheel: +-10 | Ctrl+Wheel: +-30 | R: Reset"
        cv2.putText(
            info_panel,
            controls_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

        # Add labels
        label_rgb = "RGB"
        label_depth = "Depth (Colorized)"

        cv2.putText(
            display,
            label_rgb,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            label_depth,
            (new_width + 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

        # Combine
        final_display = np.vstack([display, info_panel])

        return final_display, True

    def _on_hue_trackbar_change(self, trackbar_value: int) -> None:
        """
        Callback for hue trackbar changes

        Args:
            trackbar_value: Trackbar value (0-360, where 180 = no adjustment)
        """
        # Convert trackbar value (0-360) to hue delta (-180 to +180)
        self.current_hue_delta = trackbar_value - 180

    def _on_mouse_event(self, event, x, y, flags, param):
        """
        Callback for mouse events (wheel scrolling for hue adjustment)

        Args:
            event: OpenCV mouse event type
            x, y: Mouse position
            flags: Additional flags (wheel delta, Ctrl key, etc.)
            param: Additional parameters (window name)
        """
        window_name = param

        if event == cv2.EVENT_MOUSEWHEEL:
            # Get wheel direction (positive = up, negative = down)
            delta = cv2.getMouseWheelDelta(flags)

            # Determine adjustment amount based on Ctrl key
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # Ctrl + Wheel: ±30 degrees
                adjustment = 30 if delta > 0 else -30
            else:
                # Wheel only: ±10 degrees
                adjustment = 10 if delta > 0 else -10

            # Apply adjustment with bounds
            self.current_hue_delta = max(-180, min(180, self.current_hue_delta + adjustment))

            # Update trackbar
            cv2.setTrackbarPos("Hue Adjust", window_name, self.current_hue_delta + 180)

            logger.info(f"Hue adjusted: {self.current_hue_delta:+d} (Mouse wheel: {adjustment:+d})")

    def run_interactive_selection(self, df: pd.DataFrame) -> List[int]:
        """
        Run interactive frame selection

        Args:
            df: Metadata DataFrame

        Returns:
            List of selected frame indices
        """
        logger.info("Starting interactive selection")
        logger.info("Controls: Y=Accept, N=Reject, B=Back, Q=Quit, R=Reset Hue")
        logger.info("Hue Adjust: Arrows(+-5), [](+-30), Wheel(+-10), Ctrl+Wheel(+-30)")
        logger.info("=" * 60)

        total_frames = len(df)
        window_name = "Manual Frame Selection"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)

        # Create hue adjustment trackbar
        # Trackbar range: 0-360 (180 = no adjustment, 0 = -180, 360 = +180)
        cv2.createTrackbar("Hue Adjust", window_name, 180, 360, self._on_hue_trackbar_change)

        # Set mouse callback for wheel events (pass window_name as param)
        cv2.setMouseCallback(window_name, self._on_mouse_event, window_name)

        self.current_index = 0
        need_redraw = True  # Flag to control image redrawing

        while self.current_index < total_frames:
            # Load hue adjustment for current frame if exists
            if self.current_index in self.hue_adjustments:
                self.current_hue_delta = self.hue_adjustments[self.current_index]
                # Update trackbar position
                cv2.setTrackbarPos("Hue Adjust", window_name, self.current_hue_delta + 180)
            else:
                # Reset to no adjustment for new frame
                self.current_hue_delta = 0
                cv2.setTrackbarPos("Hue Adjust", window_name, 180)

            row = df.iloc[self.current_index]

            # Build paths
            rgb_path = self.base_dir / row['rgb_path']
            depth_path = self.base_dir / row['depth_path']

            # Frame info
            progress = f"Frame {self.current_index + 1}/{total_frames}"
            frame_id = f"ID: {row['frame_id']}"
            time_diff = f"Time Diff: {row['time_diff_ms']:.2f}ms"

            frame_info = f"{progress} | {frame_id} | {time_diff}"

            # Create and show display
            need_redraw = True
            self._last_displayed_hue = None  # Reset for new frame

            # Wait for key press with trackbar support
            while True:
                # Redraw if needed (trackbar change or first display)
                if need_redraw:
                    display, success = self.create_display_image(
                        rgb_path,
                        depth_path,
                        frame_info
                    )

                    if not success:
                        logger.warning(f"Skipping frame {self.current_index} due to load error")
                        break

                    cv2.imshow(window_name, display)
                    need_redraw = False
                    self._last_displayed_hue = self.current_hue_delta  # Track what we just displayed

                # Wait for key press (short delay for trackbar responsiveness)
                # Use waitKeyEx to detect arrow keys
                key_ex = cv2.waitKeyEx(50)
                key = key_ex & 0xFF

                # Handle arrow keys for hue adjustment
                # Arrow key codes (platform-dependent, but these are common)
                LEFT_ARROW = 2424832  # 0x250000
                RIGHT_ARROW = 2555904  # 0x270000

                hue_changed = False

                if key_ex == LEFT_ARROW or key == 81:  # Left arrow (decrease hue)
                    self.current_hue_delta = max(-180, self.current_hue_delta - 5)
                    cv2.setTrackbarPos("Hue Adjust", window_name, self.current_hue_delta + 180)
                    hue_changed = True
                    logger.info(f"Hue adjusted: {self.current_hue_delta:+d} (Left arrow: -5)")

                elif key_ex == RIGHT_ARROW or key == 83:  # Right arrow (increase hue)
                    self.current_hue_delta = min(180, self.current_hue_delta + 5)
                    cv2.setTrackbarPos("Hue Adjust", window_name, self.current_hue_delta + 180)
                    hue_changed = True
                    logger.info(f"Hue adjusted: {self.current_hue_delta:+d} (Right arrow: +5)")

                elif key == ord('['):  # '[' key for larger decrease
                    self.current_hue_delta = max(-180, self.current_hue_delta - 30)
                    cv2.setTrackbarPos("Hue Adjust", window_name, self.current_hue_delta + 180)
                    hue_changed = True
                    logger.info(f"Hue adjusted: {self.current_hue_delta:+d} ('[' key: -30)")

                elif key == ord(']'):  # ']' key for larger increase
                    self.current_hue_delta = min(180, self.current_hue_delta + 30)
                    cv2.setTrackbarPos("Hue Adjust", window_name, self.current_hue_delta + 180)
                    hue_changed = True
                    logger.info(f"Hue adjusted: {self.current_hue_delta:+d} (']' key: +30)")

                # If hue was changed by keyboard, trigger redraw
                if hue_changed:
                    need_redraw = True
                    continue

                # Check if trackbar changed (force redraw)
                current_trackbar_value = cv2.getTrackbarPos("Hue Adjust", window_name)
                expected_hue_delta = current_trackbar_value - 180
                # Trackbar callback already updates self.current_hue_delta
                # Just check if value changed and force redraw
                if expected_hue_delta != self.current_hue_delta:
                    # This shouldn't happen if callback works, but keep as safety
                    logger.warning(f"Trackbar callback missed! Manually updating: {expected_hue_delta}")
                    self.current_hue_delta = expected_hue_delta
                # Always redraw after waitKey to catch callback changes
                # Compare with last displayed value to detect changes
                if self._last_displayed_hue != self.current_hue_delta:
                    need_redraw = True
                    continue

                # Skip if no key pressed
                if key == 255:
                    continue

                if key == ord('y') or key == ord('Y'):
                    # Accept frame and save hue adjustment
                    self.selected_frames.append(self.current_index)
                    self.hue_adjustments[self.current_index] = self.current_hue_delta
                    logger.info(f"Frame {self.current_index} ACCEPTED (Total: {len(self.selected_frames)}, Hue: {self.current_hue_delta:+d})")
                    self.current_index += 1
                    break

                elif key == ord('n') or key == ord('N'):
                    # Reject frame
                    self.rejected_frames.append(self.current_index)
                    logger.info(f"Frame {self.current_index} REJECTED")
                    self.current_index += 1
                    break

                elif key == ord('r') or key == ord('R'):
                    # Reset hue to 0
                    self.current_hue_delta = 0
                    cv2.setTrackbarPos("Hue Adjust", window_name, 180)
                    need_redraw = True
                    logger.info(f"Hue reset to 0 for frame {self.current_index}")

                elif key == ord('b') or key == ord('B'):
                    # Go back
                    if self.current_index > 0:
                        self.current_index -= 1

                        # Remove from selected/rejected if present
                        if self.current_index in self.selected_frames:
                            self.selected_frames.remove(self.current_index)
                        if self.current_index in self.rejected_frames:
                            self.rejected_frames.remove(self.current_index)
                        # Keep hue adjustment for back navigation

                        logger.info(f"Going back to frame {self.current_index}")
                    break

                elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
                    # Quit
                    logger.info("User quit selection")
                    cv2.destroyAllWindows()
                    return self.selected_frames

        cv2.destroyAllWindows()

        logger.info("=" * 60)
        logger.info(f"Selection completed")
        logger.info(f"Total reviewed: {total_frames}")
        logger.info(f"Accepted: {len(self.selected_frames)}")
        logger.info(f"Rejected: {len(self.rejected_frames)}")

        return self.selected_frames

    def copy_selected_frames(
        self,
        df: pd.DataFrame,
        selected_indices: List[int]
    ) -> pd.DataFrame:
        """
        Copy selected frames to output directory

        Args:
            df: Full metadata DataFrame
            selected_indices: List of selected frame indices

        Returns:
            DataFrame with selected frames metadata
        """
        logger.info("=" * 60)
        logger.info("Copying selected frames")
        logger.info("=" * 60)

        selected_df = df.iloc[selected_indices].copy().reset_index(drop=True)

        success_count = 0
        error_count = 0

        for idx, row in selected_df.iterrows():
            try:
                # Get original index from selected_indices
                original_idx = selected_indices[idx]

                # Source paths
                src_rgb = self.base_dir / row['rgb_path']
                src_depth = self.base_dir / row['depth_path']

                # New filenames with sequential numbering
                new_filename_rgb = f"{idx:06d}.png"
                new_filename_depth = f"{idx:06d}.png"

                # Destination paths
                dst_rgb = self.output_rgb_dir / new_filename_rgb
                dst_depth = self.output_depth_dir / new_filename_depth

                # Load and apply hue adjustment to RGB
                rgb_image = cv2.imread(str(src_rgb))
                if rgb_image is not None and original_idx in self.hue_adjustments:
                    hue_delta = self.hue_adjustments[original_idx]
                    rgb_image = self.apply_hue_adjustment(rgb_image, hue_delta)
                    cv2.imwrite(str(dst_rgb), rgb_image)
                else:
                    # No hue adjustment, just copy
                    shutil.copy2(src_rgb, dst_rgb)

                # Copy depth (no adjustment)
                shutil.copy2(src_depth, dst_depth)

                # Update paths and hue adjustment in dataframe
                selected_df.at[idx, 'rgb_path'] = f"rgb/{new_filename_rgb}"
                selected_df.at[idx, 'depth_path'] = f"depth/{new_filename_depth}"
                selected_df.at[idx, 'new_index'] = idx
                selected_df.at[idx, 'hue_adjustment'] = self.hue_adjustments.get(original_idx, 0)

                success_count += 1

            except Exception as e:
                logger.error(f"Error copying frame {row['frame_id']}: {e}")
                error_count += 1

        logger.info(f"Successfully copied {success_count} frame pairs")
        if error_count > 0:
            logger.warning(f"Failed to copy {error_count} frame pairs")

        return selected_df

    def save_metadata(self, df: pd.DataFrame, filename: str = "metadata.csv"):
        """Save metadata CSV"""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved metadata to {output_path}")

    def save_summary(
        self,
        df: pd.DataFrame,
        original_count: int,
        sampled_count: int,
        sample_interval: Optional[float],
        sample_fps: Optional[float],
        start_time: datetime,
        end_time: datetime
    ):
        """Save enhanced selection summary with comprehensive statistics"""
        summary_path = self.output_dir / "selection_summary.txt"

        with open(summary_path, 'w') as f:
            f.write("Manual Frame Selection Summary\n")
            f.write("=" * 60 + "\n\n")

            # Source information
            f.write("Source Information:\n")
            f.write(f"  Source directory: {self.base_dir}\n")
            f.write(f"  Source metadata: {self.metadata_csv}\n")
            f.write(f"  Output directory: {self.output_dir}\n\n")

            # Sampling configuration
            f.write("Sampling Configuration:\n")
            if sample_interval is not None:
                f.write(f"  Sample interval: {sample_interval} seconds\n")
            else:
                f.write(f"  Sample interval: N/A\n")

            if sample_fps is not None:
                f.write(f"  Sample FPS: {sample_fps}\n")
            else:
                f.write(f"  Sample FPS: N/A\n")

            f.write(f"  Original frames: {original_count}\n")
            f.write(f"  After sampling: {sampled_count}")
            if original_count > 0:
                f.write(f" ({sampled_count/original_count*100:.1f}%)\n")
            else:
                f.write("\n")
            f.write("\n")

            # Selection statistics
            f.write("Selection Statistics:\n")
            f.write(f"  Total frames reviewed: {self.current_index}\n")
            accepted = len(self.selected_frames)
            rejected = len(self.rejected_frames)
            reviewed = self.current_index

            if reviewed > 0:
                f.write(f"  Frames accepted: {accepted} ({accepted/reviewed*100:.1f}%)\n")
                f.write(f"  Frames rejected: {rejected} ({rejected/reviewed*100:.1f}%)\n")
            else:
                f.write(f"  Frames accepted: {accepted}\n")
                f.write(f"  Frames rejected: {rejected}\n")

            skipped = sampled_count - reviewed
            f.write(f"  Frames skipped: {skipped} (not reviewed)\n\n")

            # Selected frames statistics
            if len(df) > 0:
                f.write("Selected Frames Statistics:\n")

                # Time synchronization
                if 'time_diff_ms' in df.columns:
                    f.write(f"  Time synchronization:\n")
                    f.write(f"    Mean: {df['time_diff_ms'].mean():.3f} ms\n")
                    f.write(f"    Max: {df['time_diff_ms'].max():.3f} ms\n\n")

                # Hue adjustment
                if 'hue_adjustment' in df.columns:
                    f.write(f"  Hue adjustment:\n")
                    f.write(f"    Mean: {df['hue_adjustment'].mean():+.1f}\n")
                    f.write(f"    Min: {int(df['hue_adjustment'].min()):+d}\n")
                    f.write(f"    Max: {int(df['hue_adjustment'].max()):+d}\n")
                    frames_with_adjustment = (df['hue_adjustment'] != 0).sum()
                    f.write(f"    Frames with adjustment: {frames_with_adjustment}/{len(df)} ({frames_with_adjustment/len(df)*100:.1f}%)\n\n")

                # GPS statistics
                if all(col in df.columns for col in ['gps_latitude', 'gps_longitude', 'gps_altitude']):
                    f.write(f"  GPS:\n")
                    f.write(f"    Latitude: {df['gps_latitude'].min():.6f} ~ {df['gps_latitude'].max():.6f}\n")
                    f.write(f"    Longitude: {df['gps_longitude'].min():.6f} ~ {df['gps_longitude'].max():.6f}\n")
                    f.write(f"    Altitude: {df['gps_altitude'].min():.1f} ~ {df['gps_altitude'].max():.1f} m\n\n")

                # IMU statistics
                if all(col in df.columns for col in ['imu_accel_x', 'imu_accel_y', 'imu_accel_z', 'imu_accel_mag']):
                    f.write(f"  IMU Acceleration:\n")
                    f.write(f"    X: {df['imu_accel_x'].min():.2f} ~ {df['imu_accel_x'].max():.2f} m/s²\n")
                    f.write(f"    Y: {df['imu_accel_y'].min():.2f} ~ {df['imu_accel_y'].max():.2f} m/s²\n")
                    f.write(f"    Z: {df['imu_accel_z'].min():.2f} ~ {df['imu_accel_z'].max():.2f} m/s²\n")
                    f.write(f"    Magnitude: {df['imu_accel_mag'].min():.2f} ~ {df['imu_accel_mag'].max():.2f} m/s²\n\n")

                # Camera Hz statistics
                if all(col in df.columns for col in ['rgb_hz', 'depth_hz']):
                    f.write(f"  Camera Frequency:\n")
                    f.write(f"    RGB Hz: {df['rgb_hz'].min():.1f} ~ {df['rgb_hz'].max():.1f} Hz (mean: {df['rgb_hz'].mean():.1f})\n")
                    f.write(f"    Depth Hz: {df['depth_hz'].min():.1f} ~ {df['depth_hz'].max():.1f} Hz (mean: {df['depth_hz'].mean():.1f})\n\n")

                # Timestamp range
                if 'rgb_timestamp_ns' in df.columns:
                    first_ts = df['rgb_timestamp_ns'].iloc[0]
                    last_ts = df['rgb_timestamp_ns'].iloc[-1]
                    duration_s = (last_ts - first_ts) / 1e9

                    f.write(f"Timestamp Range:\n")
                    f.write(f"  First: {first_ts}\n")
                    f.write(f"  Last:  {last_ts}\n")
                    f.write(f"  Duration: {duration_s:.1f} seconds\n\n")

            # Execution info
            duration = end_time - start_time
            duration_minutes = int(duration.total_seconds() // 60)
            duration_seconds = int(duration.total_seconds() % 60)

            f.write("Execution Info:\n")
            f.write(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Duration: {duration_minutes} minutes {duration_seconds} seconds\n")

        logger.info(f"Saved summary to {summary_path}")

    def run(self, sample_interval: Optional[float] = None, sample_fps: Optional[float] = None):
        """Run complete manual selection pipeline with optional sampling"""
        # Record start time
        start_time = datetime.now()

        logger.info("Starting manual frame selection pipeline")
        logger.info("")

        # Load metadata
        df = self.load_metadata()
        original_count = len(df)

        # Apply sampling if requested
        if sample_interval is not None or sample_fps is not None:
            logger.info("[Sampling] Applying frame sampling...")
            df = FrameSampler.apply_metadata_sampling(
                metadata_df=df,
                sample_interval=sample_interval,
                sample_fps=sample_fps
            )
            logger.info("")

        sampled_count = len(df)

        # Interactive selection
        selected_indices = self.run_interactive_selection(df)

        if len(selected_indices) == 0:
            logger.warning("No frames were selected")
            return

        # Copy selected frames
        selected_df = self.copy_selected_frames(df, selected_indices)

        # Save metadata
        self.save_metadata(selected_df)

        # Record end time
        end_time = datetime.now()

        # Save summary with all statistics
        self.save_summary(
            df=selected_df,
            original_count=original_count,
            sampled_count=sampled_count,
            sample_interval=sample_interval,
            sample_fps=sample_fps,
            start_time=start_time,
            end_time=end_time
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("Manual selection completed successfully")
        logger.info(f"Selected {len(selected_df)} frames")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Interactive manual frame selection tool'
    )

    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory containing rgb/, depth/ folders'
    )

    parser.add_argument(
        'metadata_csv',
        type=str,
        help='Path to selected_frames_metadata.csv'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory (e.g., /media/jw/storage1/data/orchard_drive_data/dataset_01)'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=1920,
        help='Display window width (default: 1920)'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=1080,
        help='Display window height (default: 1080)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--sample-interval',
        type=float,
        default=None,
        help='Time interval in seconds for frame sampling (e.g., 1.0 = one frame per second)'
    )

    parser.add_argument(
        '--sample-fps',
        type=float,
        default=None,
        help='Target FPS for frame sampling (e.g., 5.0 = sample to 5 fps)'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create selector
    selector = ManualFrameSelector(
        base_dir=args.base_dir,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        window_width=args.width,
        window_height=args.height
    )

    # Run selection with sampling options
    try:
        selector.run(
            sample_interval=args.sample_interval,
            sample_fps=args.sample_fps
        )
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
