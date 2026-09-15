"""Static/offline standalone packaging checks for product v4."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_ROOT = PRODUCT_ROOT / "collector"
sys.path.insert(0, str(COLLECTOR_ROOT))

import config  # noqa: E402
import gripper_io  # noqa: E402


CONVERTER_SHA256 = "89240669561f46f4b6ea055ec48589de9e95627e4bdb3fc4c0e460cdb8660a7e"
PLAC0_V2_SHA256 = "cf5299908fe8871cd58533d67f1def243680c1c069b3a26c17ecc1274af45e14"


class ProductV4PackagingTests(unittest.TestCase):
    def test_product_local_collection_and_log_paths(self) -> None:
        task_root = PRODUCT_ROOT / "data_teleop" / "pick_place_object_v1"
        self.assertEqual(config.PRODUCT_ROOT, PRODUCT_ROOT)
        self.assertEqual(config.DATA_COLLECTION_DIR, task_root)
        self.assertEqual(config.RAW_DATA_DIR, task_root / "raw_data")
        self.assertEqual(config.LEROBOT_DATASETS_DIR, task_root / "datasets")
        self.assertEqual(config.LOG_DIRECTORY, PRODUCT_ROOT / "logs")

    def test_correct_model_is_product_local(self) -> None:
        self.assertEqual(config.URDF_PATH, PRODUCT_ROOT / "model" / "tb6_r5_correct.urdf")
        self.assertEqual(config.MESH_DIRECTORY, PRODUCT_ROOT / "model" / "meshes")
        self.assertTrue(config.URDF_PATH.is_file())
        self.assertTrue(config.MESH_DIRECTORY.is_dir())

    def test_tb6_and_omnipicker_runtime_are_product_local(self) -> None:
        required = (
            COLLECTOR_ROOT / "vendor" / "rpc" / "rpc_transport.py",
            COLLECTOR_ROOT / "vendor" / "rpc" / "lib" / "linux" / "x86" / "2004" / "rpc.so",
            COLLECTOR_ROOT / "vendor" / "topic" / "topic.so",
            COLLECTOR_ROOT / "vendor" / "topic" / "libprotobuf.so.32",
            COLLECTOR_ROOT / "vendor" / "omnipicker" / "libusbcan.so",
        )
        for path in required:
            self.assertTrue(path.is_file(), str(path))

    def test_no_symlink_exists_anywhere_in_product(self) -> None:
        links = [path for path in PRODUCT_ROOT.rglob("*") if path.is_symlink()]
        self.assertEqual(links, [])

    def test_runtime_sources_have_no_historical_project_dependency(self) -> None:
        forbidden = (
            "Teleoperation_TB6R5_Quest3",
            "04_feature_integration",
            "quest3_tb6_r5_omnipicker_data_product_v3",
            "08_home_button_integration",
            "TB6_R5_topic_py",
            "get_status_py",
            "OmniPicker_usb2can",
            "TB6-R5-model",
            "/home/zzw/projects/",
        )
        runtime_files = (
            *COLLECTOR_ROOT.rglob("*.py"),
            *(PRODUCT_ROOT / "converter").rglob("*.py"),
            *(PRODUCT_ROOT / "scripts").glob("*.sh"),
        )
        for path in runtime_files:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{token!r} in {path}")

    def test_collect_wrapper_is_product_local_and_complete(self) -> None:
        source = (PRODUCT_ROOT / "scripts" / "run_collect.sh").read_text(encoding="utf-8")
        self.assertIn('COLLECTOR_ROOT="${PRODUCT_ROOT}/collector"', source)
        self.assertIn('"${COLLECTOR_ROOT}/teleop.py"', source)
        self.assertIn('TOPIC_DIR="${COLLECTOR_ROOT}/vendor/topic"', source)
        self.assertIn('RPC_SO_DIR="${COLLECTOR_ROOT}/vendor/rpc/lib/linux/x86/2004"', source)
        self.assertNotIn("LD_PRELOAD", source)
        for flag in (
            "--execute-motion", "--confirm-controller-ready",
            "--confirm-robot-enabled", "--confirm-robot-stationary", "--record-data",
        ):
            self.assertIn(flag, source)
        self.assertIn("STRICT INCREMENTAL / NO-BACKLOG", source)

    def test_convert_wrapper_uses_only_product_converter_and_config_paths(self) -> None:
        source = (PRODUCT_ROOT / "scripts" / "run_convert.sh").read_text(encoding="utf-8")
        self.assertIn('"${PRODUCT_ROOT}/converter/convert_tb6_omnipicker_to_lerobot.py"', source)
        self.assertIn("config.RAW_DATA_DIR.resolve()", source)
        self.assertIn("config.LEROBOT_DATASETS_DIR.resolve()", source)
        self.assertIn('if [[ -e "${OUTPUT_ROOT}" ]]', source)
        self.assertNotIn("--push-to-hub", source)

    def test_converter_is_frozen_and_schema_remains_7d_at_50hz(self) -> None:
        path = PRODUCT_ROOT / "converter" / "convert_tb6_omnipicker_to_lerobot.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), CONVERTER_SHA256)
        source = path.read_text(encoding="utf-8")
        self.assertEqual(source.count('"shape": (7,)'), 2)
        self.assertIn("fps=50", source)
        for name in ("front", "top", "wrist"):
            self.assertIn(f'"observation.images.{name}"', source)
        self.assertIn('step_data["q_actual"]', source)
        self.assertIn('step_data["q_command"]', source)

    def test_verified_v2_placo_core_is_byte_identical(self) -> None:
        digest = hashlib.sha256((COLLECTOR_ROOT / "placo_ik.py").read_bytes()).hexdigest()
        self.assertEqual(digest, PLAC0_V2_SHA256)

    def test_verified_control_and_home_parameters(self) -> None:
        self.assertEqual(config.CONTROL_HZ, 50.0)
        self.assertEqual(config.DT, 0.02)
        self.assertEqual(config.JOINT_SPEED_LIMIT_RAD_S, 0.2)
        self.assertEqual(config.MAX_ALLOWED_JOINT_SPEED_RAD_S, 0.25)
        self.assertEqual((config.TRANSLATION_SCALE, config.ROTATION_SCALE), (0.8, 0.4))
        self.assertEqual(config.RELEASE_DECEL_RAD_S2, 0.8)
        self.assertEqual(config.HOME_JOINT_DEG, (90.0, -90.0, 90.0, -90.0, -90.0, 0.0))
        self.assertEqual(config.HOME_SPEED_RAD_S, 0.08)
        self.assertEqual(config.HOME_SYNC_SPEED_RAD_S, 0.05)
        self.assertEqual(config.HOME_SYNC_BYPASS_GAP_DEG, 0.2)
        self.assertEqual(config.HOME_SYNC_MAX_INITIAL_GAP_DEG, 5.0)
        self.assertEqual(config.HOME_SYNC_READY_GAP_DEG, 0.6)
        self.assertEqual(config.HOME_SYNC_MAX_PASSES, 2)

    def test_mature_peripheral_constants_are_preserved(self) -> None:
        self.assertEqual(gripper_io.READY_KEEPALIVE_S, 0.25)
        self.assertEqual(
            (config.GRIPPER_VEL, config.GRIPPER_FORCE, config.GRIPPER_ACC, config.GRIPPER_DEC),
            (60.0, 70.0, 40.0, 40.0),
        )
        self.assertEqual(config.CAMERA_FPS, 60)
        self.assertEqual(config.CAMERA_JPG_QUALITY, 85)
        self.assertEqual(config.DATA_LOG_HZ, 50.0)
        self.assertEqual(config.CAMERA_PREVIEW_HZ, 20.0)

    def test_training_entry_schema_and_accepted_command_source_are_frozen(self) -> None:
        source = (COLLECTOR_ROOT / "teleop.py").read_text(encoding="utf-8")
        expected = (
            '"timestamp"', '"step_index"', '"q_actual"', '"q_command"',
            '"gripper_actual_pos"', '"gripper_command_pos"', '"image"',
            '"topic_global_count"',
        )
        for field in expected:
            self.assertIn(field, source)
        self.assertIn(
            "recorder.record(started, topic, transport.last_accepted_command_q, snapshot)",
            source,
        )

    def test_formal_product_identity_and_log_name_are_v4(self) -> None:
        source = (COLLECTOR_ROOT / "teleop.py").read_text(encoding="utf-8")
        self.assertIn("quest3_tb6_r5_omnipicker_data_product_v4 READY", source)
        self.assertIn('f"product_v4_', source)
        self.assertNotIn("feature_integration_", source)

    def test_manifest_excludes_mutable_runtime_data(self) -> None:
        manifest = (PRODUCT_ROOT / "MANIFEST.sha256").read_text(encoding="utf-8")
        self.assertNotIn("logs/", manifest)
        self.assertNotIn("raw_data/", manifest)
        self.assertNotIn("datasets/", manifest)
        self.assertNotIn("__pycache__", manifest)
        self.assertIn("collector/teleop.py", manifest)
        self.assertIn("model/tb6_r5_correct.urdf", manifest)
        self.assertIn("converter/convert_tb6_omnipicker_to_lerobot.py", manifest)


if __name__ == "__main__":
    unittest.main()
