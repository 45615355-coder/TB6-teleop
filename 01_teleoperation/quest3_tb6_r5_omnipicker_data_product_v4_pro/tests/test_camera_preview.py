"""Offline tests for the non-critical latest-JPEG camera preview."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_ROOT = PRODUCT_ROOT / "collector"
sys.path.insert(0, str(COLLECTOR_ROOT))

import config  # noqa: E402
from camera_preview import CameraPreview  # noqa: E402


SERIALS = {"front": "front-serial", "top": "top-serial", "wrist": "wrist-serial"}


class MockCamera:
    def __init__(self, frames: dict[str, object]) -> None:
        self.frames = frames
        self.read_count = 0

    def get_compressed_frames(self) -> dict[str, object]:
        self.read_count += 1
        return self.frames


def jpeg(color_bgr: tuple[int, int, int]) -> bytes:
    image = np.empty((480, 640, 3), dtype=np.uint8)
    image[:] = color_bgr
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("offline JPEG encoding failed")
    return encoded.tobytes()


def complete_frames() -> dict[str, dict[str, object]]:
    return {
        SERIALS["front"]: {"color": jpeg((255, 0, 0)), "frame_number": 1},
        SERIALS["top"]: {"color": jpeg((0, 255, 0)), "frame_number": 2},
        SERIALS["wrist"]: {"color": jpeg((0, 0, 255)), "frame_number": 3},
    }


def oriented_jpeg(seed: int) -> bytes:
    y, x = np.indices((480, 640))
    image = np.empty((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = (x + seed) % 256
    image[:, :, 1] = (y + 2 * seed) % 256
    image[:, :, 2] = ((x // 3) + (y // 2) + 3 * seed) % 256
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("offline oriented JPEG encoding failed")
    return encoded.tobytes()


def decode(jpeg_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("offline JPEG decoding failed")
    return image


class CameraPreviewTests(unittest.TestCase):
    def make_preview(self, frames: dict[str, object] | None = None) -> CameraPreview:
        return CameraPreview(
            camera=MockCamera({} if frames is None else frames),
            camera_serial_dict=SERIALS,
            hz=config.CAMERA_PREVIEW_HZ,
            width=config.CAMERA_PREVIEW_WIDTH,
            height=config.CAMERA_PREVIEW_HEIGHT,
            window_name=config.CAMERA_PREVIEW_WINDOW_NAME,
        )

    def test_module_has_no_realsense_owner_or_hardware_imports(self) -> None:
        source = (COLLECTOR_ROOT / "camera_preview.py").read_text(encoding="utf-8")
        self.assertNotIn("pyrealsense2", source)
        self.assertNotIn("RealSenseCameraInterface", source)
        self.assertNotIn("update_frames", source)
        self.assertNotIn("camera.start", source)

    def test_three_mock_jpegs_decode_in_front_top_wrist_order(self) -> None:
        canvas = self.make_preview(complete_frames()).compose_canvas(complete_frames())
        self.assertIsNotNone(canvas)
        assert canvas is not None
        self.assertEqual(
            canvas.shape,
            (config.CAMERA_PREVIEW_HEIGHT * 3, config.CAMERA_PREVIEW_WIDTH, 3),
        )
        x = config.CAMERA_PREVIEW_WIDTH // 2
        centers = [
            canvas[config.CAMERA_PREVIEW_HEIGHT // 2, x],
            canvas[config.CAMERA_PREVIEW_HEIGHT + config.CAMERA_PREVIEW_HEIGHT // 2, x],
            canvas[2 * config.CAMERA_PREVIEW_HEIGHT + config.CAMERA_PREVIEW_HEIGHT // 2, x],
        ]
        self.assertEqual([int(np.argmax(pixel)) for pixel in centers], [0, 1, 2])

    def test_only_top_rotates_90_degrees_clockwise(self) -> None:
        frames = {
            SERIALS["front"]: {"color": oriented_jpeg(7), "frame_number": 1},
            SERIALS["top"]: {"color": oriented_jpeg(31), "frame_number": 2},
            SERIALS["wrist"]: {"color": oriented_jpeg(73), "frame_number": 3},
        }
        preview = self.make_preview(frames)
        front = decode(frames[SERIALS["front"]]["color"])
        top = decode(frames[SERIALS["top"]]["color"])
        wrist = decode(frames[SERIALS["wrist"]]["color"])
        expected = np.vstack([
            preview._resize_complete_frame(front),
            preview._resize_complete_frame(cv2.rotate(top, cv2.ROTATE_90_CLOCKWISE)),
            preview._resize_complete_frame(wrist),
        ])

        with patch(
            "camera_preview.cv2.putText",
            side_effect=lambda image, *args, **kwargs: image,
        ), patch("camera_preview.cv2.rotate", wraps=cv2.rotate) as rotate:
            canvas = preview.compose_canvas(frames)

        rotate.assert_called_once()
        self.assertEqual(rotate.call_args.args[1], cv2.ROTATE_90_CLOCKWISE)
        self.assertIsNotNone(canvas)
        assert canvas is not None
        np.testing.assert_array_equal(canvas, expected)
        self.assertEqual(
            canvas.shape,
            (config.CAMERA_PREVIEW_HEIGHT * 3, config.CAMERA_PREVIEW_WIDTH, 3),
        )

    def test_incomplete_bundle_is_skipped_without_placeholder(self) -> None:
        frames = complete_frames()
        del frames[SERIALS["top"]]
        self.assertIsNone(self.make_preview(frames).compose_canvas(frames))

    def test_preview_reads_only_latest_compressed_cache(self) -> None:
        source = (COLLECTOR_ROOT / "camera_preview.py").read_text(encoding="utf-8")
        self.assertIn("self._camera.get_compressed_frames()", source)
        self.assertNotIn("FIFO", source)

    def test_preview_failure_does_not_set_recording_stop(self) -> None:
        preview_source = (COLLECTOR_ROOT / "camera_preview.py").read_text(encoding="utf-8")
        teleop_source = (COLLECTOR_ROOT / "teleop.py").read_text(encoding="utf-8")
        preview_startup = teleop_source[
            teleop_source.index("if config.CAMERA_PREVIEW_ENABLED:"):
            teleop_source.index("return EpisodeRecorder", teleop_source.index("if config.CAMERA_PREVIEW_ENABLED:"))
        ]
        self.assertNotIn(".set()", preview_source)
        self.assertNotIn("recording_stop.set()", preview_startup)
        self.assertIn("[PREVIEW] stopped:", preview_source)

    def test_preview_rate_is_20_hz(self) -> None:
        self.assertEqual(config.CAMERA_PREVIEW_HZ, 20.0)


if __name__ == "__main__":
    unittest.main()
