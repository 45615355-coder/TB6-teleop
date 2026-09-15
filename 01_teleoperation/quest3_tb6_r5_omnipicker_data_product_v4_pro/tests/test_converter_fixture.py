"""One-step frozen-converter smoke test with a fake LeRobot dataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pickle
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = PRODUCT_ROOT / "converter" / "convert_tb6_omnipicker_to_lerobot.py"


class FakeDataset:
    latest: "FakeDataset | None" = None

    def __init__(self, **create_kwargs) -> None:
        self.create_kwargs = create_kwargs
        self.frames: list[dict[str, object]] = []
        self.saved = 0
        self.finalized = False
        FakeDataset.latest = self

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def add_frame(self, frame) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved += 1

    def finalize(self) -> None:
        self.finalized = True


class ConverterFixtureTests(unittest.TestCase):
    def test_current_pkl_schema_converts_to_frozen_7d_observation_and_action(self) -> None:
        package = ModuleType("lerobot")
        package.__path__ = []
        datasets = ModuleType("lerobot.datasets")
        datasets.__path__ = []
        dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
        dataset_module.LeRobotDataset = FakeDataset
        with patch.dict(sys.modules, {
            "lerobot": package,
            "lerobot.datasets": datasets,
            "lerobot.datasets.lerobot_dataset": dataset_module,
        }):
            spec = importlib.util.spec_from_file_location("_product_v4_converter_fixture", CONVERTER_PATH)
            assert spec is not None and spec.loader is not None
            converter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(converter)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        cameras = {
            name: {"color": encoded.tobytes(), "timestamp": 1.0, "frame_number": 1}
            for name in ("front", "top", "wrist")
        }
        entry = {
            "timestamp": 1.0,
            "step_index": 0,
            "q_actual": np.arange(6, dtype=float),
            "q_command": np.arange(10, 16, dtype=float),
            "gripper_actual_pos": 17.0,
            "gripper_command_pos": 83.0,
            "image": cameras,
            "topic_global_count": 123,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"; raw.mkdir()
            with (raw / "episode.pkl").open("wb") as file:
                pickle.dump([entry], file)
            converter.parse_args = lambda: SimpleNamespace(
                data_dir=raw,
                output_root=root / "dataset",
                repo_id="local/test",
                task="fixture task",
                num_episodes=None,
            )
            converter.main()

        dataset = FakeDataset.latest
        assert dataset is not None
        self.assertEqual(dataset.create_kwargs["fps"], 50)
        self.assertEqual(dataset.saved, 1)
        self.assertTrue(dataset.finalized)
        self.assertEqual(len(dataset.frames), 1)
        frame = dataset.frames[0]
        np.testing.assert_array_equal(frame["observation.state"],
                                      np.array((0, 1, 2, 3, 4, 5, 17), dtype=np.float32))
        np.testing.assert_array_equal(frame["action"],
                                      np.array((10, 11, 12, 13, 14, 15, 83), dtype=np.float32))
        for name in ("front", "top", "wrist"):
            self.assertEqual(frame[f"observation.images.{name}"].shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
