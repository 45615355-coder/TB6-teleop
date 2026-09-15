"""Non-critical GUI preview fed only by the existing compressed-frame cache."""

from __future__ import annotations

import time
from typing import Any, Mapping

import cv2
import numpy as np


VIEW_ORDER = ("front", "top", "wrist")


class CameraPreview:
    """Display the latest front/top/wrist JPEG bundle in one vertical window."""

    def __init__(
        self,
        *,
        camera: Any,
        camera_serial_dict: Mapping[str, str],
        hz: float,
        width: int,
        height: int,
        window_name: str,
    ) -> None:
        if camera is None or not callable(getattr(camera, "get_compressed_frames", None)):
            raise ValueError("camera must expose get_compressed_frames()")
        if any(view not in camera_serial_dict for view in VIEW_ORDER):
            raise ValueError("camera_serial_dict must contain front/top/wrist")
        if hz <= 0.0 or width <= 0 or height <= 0 or not window_name:
            raise ValueError("preview hz, dimensions, and window name must be valid")
        self._camera = camera
        self._serials = {view: str(camera_serial_dict[view]) for view in VIEW_ORDER}
        self._period_s = 1.0 / float(hz)
        self._width = int(width)
        self._height = int(height)
        self._window_name = str(window_name)

    def compose_canvas(self, frames: Mapping[str, Any]) -> np.ndarray | None:
        """Decode one complete latest bundle, or skip an incomplete bundle."""
        images: list[np.ndarray] = []
        for view in VIEW_ORDER:
            frame = frames.get(self._serials[view])
            if frame is None or "color" not in frame:
                return None
            jpeg_bytes = frame["color"]
            if not isinstance(jpeg_bytes, (bytes, bytearray, memoryview)):
                return None
            encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not decode {view} JPEG")
            if view == "top":
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image = self._resize_complete_frame(image)
            frame_number = frame.get("frame_number")
            label = view.upper()
            if frame_number is not None:
                label += f"  #{frame_number}"
            cv2.putText(
                image,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            images.append(image)
        return np.vstack(images)

    def _resize_complete_frame(self, image: np.ndarray) -> np.ndarray:
        """Fit the whole image into the configured box without crop or stretch."""
        source_height, source_width = image.shape[:2]
        scale = min(self._width / source_width, self._height / source_height)
        resized_width = max(1, min(self._width, round(source_width * scale)))
        resized_height = max(1, min(self._height, round(source_height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        x = (self._width - resized_width) // 2
        y = (self._height - resized_height) // 2
        canvas[y:y + resized_height, x:x + resized_width] = resized
        return canvas

    def run(self, stop_event: Any) -> None:
        """Run a latest-only GUI loop; failures remain isolated to this thread."""
        next_tick = time.monotonic()
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            while not stop_event.is_set():
                frames = self._camera.get_compressed_frames()
                canvas = self.compose_canvas(frames)
                if canvas is not None:
                    cv2.imshow(self._window_name, canvas)
                cv2.waitKey(1)
                if cv2.getWindowProperty(
                    self._window_name, cv2.WND_PROP_VISIBLE
                ) < 1.0:
                    raise RuntimeError("preview window was closed")

                next_tick += self._period_s
                delay = next_tick - time.monotonic()
                if delay > 0.0:
                    stop_event.wait(delay)
                else:
                    next_tick = time.monotonic()
        except Exception as error:
            print(f"[PREVIEW] stopped: {type(error).__name__}: {error}", flush=True)
        finally:
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
