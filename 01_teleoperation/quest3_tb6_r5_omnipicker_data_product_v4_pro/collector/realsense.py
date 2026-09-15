"""Minimal RGB-only capture interface for three Intel RealSense D435 cameras."""

import threading

import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseCameraInterface:
    """Capture complete RGB bundles from RealSense cameras in a fixed order."""

    def __init__(
        self,
        width=640,
        height=480,
        fps=60,
        serial_numbers=None,
        enable_compression=True,
        jpg_quality=85,
    ):
        if not serial_numbers:
            raise ValueError("serial_numbers must list the three RealSense serial numbers.")
        if len(serial_numbers) != 3 or len(set(serial_numbers)) != 3:
            raise ValueError("serial_numbers must contain exactly three unique serial numbers.")
        if not 1 <= jpg_quality <= 100:
            raise ValueError("jpg_quality must be between 1 and 100.")

        self.width = width
        self.height = height
        self.fps = fps
        self.serial_numbers = list(serial_numbers)
        self.enable_compression = enable_compression
        self.jpg_quality = jpg_quality
        self.pipelines = {}
        self.configs = {}
        self.frames_dict = {}
        self.compressed_frames_dict = {}
        self.frames_lock = threading.Lock()
        self._started = False

        context = rs.context()
        devices = context.query_devices()
        connected_serials = {
            device.get_info(rs.camera_info.serial_number) for device in devices
        }
        missing_serials = [
            serial for serial in self.serial_numbers if serial not in connected_serials
        ]
        if missing_serials:
            raise RuntimeError(
                "Requested RealSense device(s) not found: " + ", ".join(missing_serials)
            )

        for serial in self.serial_numbers:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.rgb8,
                self.fps,
            )
            self.pipelines[serial] = pipeline
            self.configs[serial] = config

    def start(self):
        """Start all RGB-only pipelines in serial_numbers order."""
        if self._started:
            return

        started_pipelines = []
        try:
            for serial in self.serial_numbers:
                self.pipelines[serial].start(self.configs[serial])
                started_pipelines.append(self.pipelines[serial])
        except Exception:
            for pipeline in reversed(started_pipelines):
                pipeline.stop()
            raise
        self._started = True

    def update_frames(self, timeout_ms=1000):
        """Read one complete bundle, waiting for cameras strictly in fixed order.

        Returns True only when all three RGB frames were read and published together.
        A failed or missing frame leaves the last complete bundle unchanged.
        """
        if not self._started:
            raise RuntimeError("Call start() before update_frames().")

        frames_dict = {}
        for serial in self.serial_numbers:
            try:
                frames = self.pipelines[serial].wait_for_frames(timeout_ms=timeout_ms)
            except RuntimeError:
                return False

            color_frame = frames.get_color_frame()
            if not color_frame:
                return False

            frames_dict[serial] = {
                "color": np.asanyarray(color_frame.get_data()).copy(),
                "timestamp": color_frame.get_timestamp(),
                "frame_number": color_frame.get_frame_number(),
            }

        compressed_frames_dict = {}
        if self.enable_compression:
            for serial, frame_data in frames_dict.items():
                compressed_frames_dict[serial] = {
                    "color": self._encode_jpeg_rgb(frame_data["color"]),
                    "timestamp": frame_data["timestamp"],
                    "frame_number": frame_data["frame_number"],
                }

        # Publish raw and compressed data as one completed camera update.
        with self.frames_lock:
            self.frames_dict = frames_dict
            self.compressed_frames_dict = compressed_frames_dict
        return True

    def get_frames(self):
        """Return the latest complete raw RGB bundle."""
        with self.frames_lock:
            return {serial: frame.copy() for serial, frame in self.frames_dict.items()}

    def get_compressed_frames(self):
        """Return the latest complete JPEG RGB bundle, or an empty dict if disabled."""
        with self.frames_lock:
            return {
                serial: frame.copy()
                for serial, frame in self.compressed_frames_dict.items()
            }

    def stop(self):
        """Stop all previously started pipelines."""
        if not self._started:
            return
        for serial in self.serial_numbers:
            try:
                self.pipelines[serial].stop()
            except RuntimeError:
                pass
        self._started = False

    def _encode_jpeg_rgb(self, color_rgb):
        """Encode RGB data as JPEG while accounting for OpenCV's BGR convention."""
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", color_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
        )
        if not ok:
            raise RuntimeError("OpenCV could not JPEG-encode a RealSense RGB frame.")
        return encoded.tobytes()
