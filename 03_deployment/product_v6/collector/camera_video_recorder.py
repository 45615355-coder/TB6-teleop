"""Non-critical latest-frame 1x3 MP4 recorder for the existing camera cache."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


VIEW_ORDER = ("front", "top", "wrist")


class CameraVideoRecorder:
    """Write FRONT | TOP | WRIST without owning or updating any camera."""

    def __init__(
        self,
        *,
        camera: Any,
        camera_serial_dict: Mapping[str, str],
        fps: float,
        panel_width: int,
        panel_height: int,
        output_directory: Path,
        run_stamp: str,
    ) -> None:
        if camera is None or not callable(getattr(camera, "get_frames", None)):
            raise ValueError("camera must expose get_frames()")
        if any(view not in camera_serial_dict for view in VIEW_ORDER):
            raise ValueError("camera_serial_dict must contain front/top/wrist")
        if fps <= 0.0 or panel_width <= 0 or panel_height <= 0:
            raise ValueError("video fps and panel dimensions must be positive")
        if not run_stamp:
            raise ValueError("run_stamp must not be empty")

        self._camera = camera
        self._serials = {view: str(camera_serial_dict[view]) for view in VIEW_ORDER}
        self._fps = float(fps)
        self._period_s = 1.0 / self._fps
        self._panel_width = int(panel_width)
        self._panel_height = int(panel_height)
        self._frame_size = (self._panel_width * len(VIEW_ORDER), self._panel_height)
        self._output_directory = Path(output_directory)
        self._output_path = self._output_directory / f"act_real_full_{run_stamp}_1x3.mp4"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames_written = 0

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def start(self) -> None:
        """Start the independent latest-only recorder thread."""
        if self._thread is not None:
            raise RuntimeError("camera video recorder has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="camera-video-recorder",
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        """Request a prompt stop without blocking safety cleanup."""
        self._stop_event.set()

    def stop(self) -> None:
        """Stop and wait until VideoWriter has been released."""
        self.request_stop()
        if self._thread is not None:
            self._thread.join()

    def compose_canvas(self, frames: Mapping[str, Any]) -> np.ndarray | None:
        """Copy and compose one complete raw-RGB bundle, skipping incomplete data."""
        rgb_images: list[tuple[str, np.ndarray]] = []
        for view in VIEW_ORDER:
            frame = frames.get(self._serials[view])
            if not isinstance(frame, Mapping) or "color" not in frame:
                return None
            color = frame["color"]
            if (
                not isinstance(color, np.ndarray)
                or color.ndim != 3
                or color.shape[2] != 3
                or color.dtype != np.uint8
                or color.shape[:2] != (self._panel_height, self._panel_width)
            ):
                return None
            # The RealSense cache lock has already been released by get_frames().
            # Copy every needed array before doing any conversion or encoding work.
            rgb_images.append((view, np.ascontiguousarray(color.copy())))

        panels: list[np.ndarray] = []
        for view, color_rgb in rgb_images:
            panel = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                panel,
                view.upper(),
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
        return np.hstack(panels)

    def _run(self) -> None:
        writer: Any = None
        writer_opened = False
        release_ok = True
        try:
            while not self._stop_event.is_set():
                cycle_started = time.monotonic()
                frames = self._camera.get_frames()
                canvas = self.compose_canvas(frames)
                if canvas is not None:
                    if writer is None:
                        self._output_directory.mkdir(parents=True, exist_ok=True)
                        if self._output_path.exists():
                            raise FileExistsError(
                                f"refusing to overwrite existing video: {self._output_path}"
                            )
                        writer = cv2.VideoWriter(
                            str(self._output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            self._fps,
                            self._frame_size,
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"could not open MP4 VideoWriter: {self._output_path}"
                            )
                        writer_opened = True
                        print(f"[VIDEO] RECORDING | path={self._output_path}", flush=True)
                    writer.write(canvas)
                    self._frames_written += 1

                remaining = self._period_s - (time.monotonic() - cycle_started)
                if remaining > 0.0:
                    self._stop_event.wait(remaining)
        except Exception as error:
            print(
                f"[VIDEO] WARNING | stopped: {type(error).__name__}: {error}",
                flush=True,
            )
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception as error:
                    release_ok = False
                    print(
                        f"[VIDEO] WARNING | release failed: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                if writer_opened and release_ok and self._frames_written > 0:
                    print(
                        f"[VIDEO] SAVED | path={self._output_path} | "
                        f"frames={self._frames_written}",
                        flush=True,
                    )
