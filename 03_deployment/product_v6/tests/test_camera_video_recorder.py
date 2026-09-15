"""Offline tests for the non-critical latest-raw-frame MP4 recorder."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_ROOT = PRODUCT_ROOT / "collector"
sys.path.insert(0, str(COLLECTOR_ROOT))

import config  # noqa: E402
from camera_video_recorder import CameraVideoRecorder  # noqa: E402


SERIALS = {"front": "front-serial", "top": "top-serial", "wrist": "wrist-serial"}


class MockCamera:
    def __init__(self, frames: dict[str, object] | None = None) -> None:
        self.frames = {} if frames is None else frames
        self.read_count = 0
        self.on_get = None

    def get_frames(self) -> dict[str, object]:
        self.read_count += 1
        if self.on_get is not None:
            self.on_get()
        return self.frames


def recorder(
    camera: MockCamera | None = None,
    output_directory: Path | None = None,
) -> CameraVideoRecorder:
    return CameraVideoRecorder(
        camera=MockCamera() if camera is None else camera,
        camera_serial_dict=SERIALS,
        fps=config.CAMERA_VIDEO_FPS,
        panel_width=config.CAMERA_VIDEO_PANEL_WIDTH,
        panel_height=config.CAMERA_VIDEO_PANEL_HEIGHT,
        output_directory=config.CAMERA_VIDEO_DIR if output_directory is None else output_directory,
        run_stamp="offline_test",
    )


def complete_frames() -> dict[str, dict[str, np.ndarray]]:
    frames = {}
    for index, view in enumerate(("front", "top", "wrist"), start=1):
        y, x = np.indices((480, 640))
        image = np.empty((480, 640, 3), dtype=np.uint8)
        image[:, :, 0] = (x + index * 11) % 256
        image[:, :, 1] = (y + index * 17) % 256
        image[:, :, 2] = ((x // 2) + (y // 3) + index * 23) % 256
        frames[SERIALS[view]] = {"color": image}
    return frames


class CameraVideoRecorderTests(unittest.TestCase):
    def test_module_has_no_realsense_owner_or_capture_calls(self) -> None:
        source = (COLLECTOR_ROOT / "camera_video_recorder.py").read_text(encoding="utf-8")
        self.assertNotIn("pyrealsense2", source)
        self.assertNotIn("RealSenseCameraInterface", source)
        self.assertNotIn("update_frames", source)
        self.assertNotIn("wait_for_frames", source)
        self.assertNotIn("cv2.rotate", source)
        self.assertEqual(source.count("self._camera.get_frames()"), 1)

    def test_canvas_is_unrotated_front_top_wrist_1920_by_480(self) -> None:
        video = recorder()
        frames = complete_frames()
        with patch(
            "camera_video_recorder.cv2.putText",
            side_effect=lambda image, *args, **kwargs: image,
        ):
            canvas = video.compose_canvas(frames)

        self.assertIsNotNone(canvas)
        assert canvas is not None
        self.assertEqual(canvas.shape, (480, 1920, 3))
        for index, view in enumerate(("front", "top", "wrist")):
            with self.subTest(view=view):
                source_rgb = frames[SERIALS[view]]["color"]
                expected = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
                np.testing.assert_array_equal(
                    canvas[:, index * 640:(index + 1) * 640],
                    expected,
                )

    def test_top_has_no_rotation_or_added_black_borders(self) -> None:
        frames = complete_frames()
        with patch(
            "camera_video_recorder.cv2.putText",
            side_effect=lambda image, *args, **kwargs: image,
        ):
            canvas = recorder().compose_canvas(frames)

        assert canvas is not None
        expected_top = cv2.cvtColor(
            frames[SERIALS["top"]]["color"], cv2.COLOR_RGB2BGR
        )
        np.testing.assert_array_equal(canvas[:, 640:1280], expected_top)

    def test_incomplete_bundle_is_skipped(self) -> None:
        frames = complete_frames()
        del frames[SERIALS["top"]]
        self.assertIsNone(recorder().compose_canvas(frames))

    def test_wrong_sized_bundle_is_skipped_without_resize_or_padding(self) -> None:
        frames = complete_frames()
        frames[SERIALS["top"]] = {
            "color": np.zeros((640, 480, 3), dtype=np.uint8)
        }
        self.assertIsNone(recorder().compose_canvas(frames))

    def test_one_get_frames_call_writes_one_latest_bundle(self) -> None:
        camera = MockCamera(complete_frames())
        with TemporaryDirectory() as temporary_directory:
            video = recorder(camera, Path(temporary_directory))
            camera.on_get = video.request_stop
            writer = MagicMock()
            writer.isOpened.return_value = True
            with patch(
                "camera_video_recorder.cv2.VideoWriter", return_value=writer
            ), patch(
                "camera_video_recorder.cv2.putText",
                side_effect=lambda image, *args, **kwargs: image,
            ):
                video._run()

        self.assertEqual(camera.read_count, 1)
        writer.write.assert_called_once()
        writer.release.assert_called_once()

    def test_overlay_contains_only_view_names(self) -> None:
        with patch("camera_video_recorder.cv2.putText") as put_text:
            canvas = recorder().compose_canvas(complete_frames())
        self.assertIsNotNone(canvas)
        self.assertEqual(
            [call.args[1] for call in put_text.call_args_list],
            ["FRONT", "TOP", "WRIST"],
        )

    def test_video_configuration_is_30_fps_and_1920_by_480(self) -> None:
        self.assertTrue(config.CAMERA_VIDEO_ENABLED)
        self.assertEqual(config.CAMERA_VIDEO_FPS, 30.0)
        self.assertEqual(config.CAMERA_VIDEO_PANEL_WIDTH, 640)
        self.assertEqual(config.CAMERA_VIDEO_PANEL_HEIGHT, 480)
        self.assertEqual(config.CAMERA_VIDEO_DIR, config.LOG_DIRECTORY / "videos")


if __name__ == "__main__":
    unittest.main()
