#!/usr/bin/env python3
"""Reconstruct and plot offline ACT action trajectories.

Inference runs in the LeRobot 0.6.1 environment and reuses the validation
helpers in evaluate_act_checkpoints.py. No robot, teleoperator, environment, or
training interface is imported. If matplotlib is unavailable in that Conda
environment, this same file is invoked in a render-only subprocess using an
already-installed local Python with matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Keep the evaluation source directory read-only during imports.
sys.dont_write_bytecode = True

# Apply offline constraints before importing LeRobot / Hugging Face code.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(Path(tempfile.gettempdir()) / "tb6_act_offline_evaluation_hf_cache"),
)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tb6_act_plot_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "tb6_act_plot_cache"))

DEFAULT_DATASET_ROOT = Path(
    "/home/zzw/projects/data_collect/product/quest3_tb6_r5_omnipicker_data_product_v4/"
    "data_teleop/pick_place_object_v1/test_datasets/pick_place_object_v1_test"
)
DEFAULT_CHECKPOINTS_ROOT = Path(
    "/home/zzw/projects/TB6_R5_ACT/training_results/pick_place_object_v1/act/v1/"
    "train/20260827_113046/checkpoints"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zzw/projects/TB6_R5_ACT/training_results/pick_place_object_v1/act/v1/"
    "evaluation_plots"
)
DEFAULT_CHECKPOINT = "080000"
DEFAULT_EPISODES = (0, 5, 7)


class TrajectoryError(RuntimeError):
    """Fatal trajectory reconstruction or rendering error."""


def parse_render_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render trajectory NPZ with matplotlib.")
    parser.add_argument("--render-npz", required=True, type=Path)
    parser.add_argument("--render-output-dir", required=True, type=Path)
    parser.add_argument("--render-checkpoint", required=True)
    parser.add_argument("--render-episode", required=True, type=int)
    return parser.parse_args(argv)


def add_chunk_boundaries(axis: Any, length: int, chunk_size: int) -> None:
    for chunk_start in range(0, length, chunk_size):
        axis.axvline(chunk_start, color="0.55", linestyle="--", linewidth=0.7, alpha=0.7)


def render_npz(argv: list[str]) -> int:
    args = parse_render_args(argv)
    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_path = args.render_npz.expanduser().resolve()
    output_dir = args.render_output_dir.expanduser().resolve()
    if not npz_path.is_file():
        raise TrajectoryError(f"Missing trajectory NPZ: {npz_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=False) as data:
        required = {"ground_truth_action", "predicted_action"}
        if not required.issubset(data.files):
            raise TrajectoryError(f"NPZ is missing required arrays {required}: {npz_path}")
        ground_truth = np.asarray(data["ground_truth_action"], dtype=np.float64)
        prediction = np.asarray(data["predicted_action"], dtype=np.float64)

    if ground_truth.ndim != 2 or ground_truth.shape[1] != 7:
        raise TrajectoryError(f"ground_truth_action shape must be [T, 7], got {ground_truth.shape}")
    if prediction.shape != ground_truth.shape:
        raise TrajectoryError(
            f"Prediction/ground-truth length mismatch: {prediction.shape} vs {ground_truth.shape}"
        )
    if not np.isfinite(ground_truth).all() or not np.isfinite(prediction).all():
        raise TrajectoryError(f"Trajectory NPZ contains NaN/Inf: {npz_path}")

    length = int(ground_truth.shape[0])
    frame_index = np.arange(length)
    ground_truth_deg = np.rad2deg(ground_truth[:, :6])
    prediction_deg = np.rad2deg(prediction[:, :6])

    figure, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
    for joint_index, axis in enumerate(axes.flat):
        axis.plot(frame_index, ground_truth_deg[:, joint_index], label="ground truth", linewidth=1.0)
        axis.plot(frame_index, prediction_deg[:, joint_index], label="prediction", linewidth=1.0)
        add_chunk_boundaries(axis, length, 100)
        axis.set_title(f"J{joint_index + 1}")
        axis.set_ylabel("action (degree)")
        axis.grid(True, linewidth=0.4, alpha=0.4)
        axis.legend(loc="best")
    axes[-1, 0].set_xlabel("frame index")
    axes[-1, 1].set_xlabel("frame index")
    figure.suptitle(
        f"ACT joint action trajectories | checkpoint={args.render_checkpoint} | "
        f"episode={args.render_episode}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    joints_path = output_dir / f"episode_{args.render_episode:03d}_joints.png"
    figure.savefig(joints_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(16, 5))
    axis.plot(frame_index, ground_truth[:, 6], label="ground truth", linewidth=1.0)
    axis.plot(frame_index, prediction[:, 6], label="prediction", linewidth=1.0)
    add_chunk_boundaries(axis, length, 100)
    axis.set_title(
        f"ACT gripper action trajectory | checkpoint={args.render_checkpoint} | "
        f"episode={args.render_episode}"
    )
    axis.set_xlabel("frame index")
    axis.set_ylabel("action (percentage point)")
    axis.grid(True, linewidth=0.4, alpha=0.4)
    axis.legend(loc="best")
    figure.tight_layout()
    gripper_path = output_dir / f"episode_{args.render_episode:03d}_gripper.png"
    figure.savefig(gripper_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    for path in (joints_path, gripper_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise TrajectoryError(f"matplotlib failed to create PNG: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline ACT trajectory reconstruction and matplotlib visualization."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoints-root", type=Path, default=DEFAULT_CHECKPOINTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(DEFAULT_EPISODES))
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument(
        "--plot-python",
        type=Path,
        default=None,
        help="Optional existing Python interpreter containing numpy and matplotlib.",
    )
    return parser.parse_args()


def import_evaluation_module() -> Any:
    module_path = Path(__file__).resolve().with_name("evaluate_act_checkpoints.py")
    if not module_path.is_file():
        raise TrajectoryError(f"Missing existing evaluation module: {module_path}")
    spec = importlib.util.spec_from_file_location("tb6_evaluate_act_checkpoints", module_path)
    if spec is None or spec.loader is None:
        raise TrajectoryError(f"Unable to import existing evaluation module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TrajectoryError(f"Failed to import existing evaluation module {module_path}: {exc}") from exc
    return module


def plot_probe_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tb6_act_plot_mplconfig"))
    environment.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "tb6_act_plot_cache"))
    return environment


def interpreter_supports_matplotlib(python_path: Path) -> bool:
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        return False
    probe = (
        "import numpy,matplotlib; matplotlib.use('Agg'); "
        "import matplotlib.pyplot; print(matplotlib.__version__)"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", probe],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            env=plot_probe_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def find_plot_python(requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested.expanduser().resolve())
    environment_python = os.environ.get("TB6_ACT_PLOT_PYTHON")
    if environment_python:
        candidates.append(Path(environment_python).expanduser().resolve())
    candidates.extend(
        [
            Path(sys.executable).resolve(),
            Path("/usr/bin/python3"),
            Path("/home/zzw/miniconda3/envs/xrteleop/bin/python"),
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if interpreter_supports_matplotlib(candidate):
            return candidate
    raise TrajectoryError(
        "No existing local Python interpreter with numpy + matplotlib was found. "
        "Nothing was installed; set --plot-python to an already-installed interpreter."
    )


def validate_requested(args: argparse.Namespace, evalmod: Any) -> tuple[str, list[int]]:
    checkpoint = args.checkpoint
    if not checkpoint.isdigit() or Path(checkpoint).name != checkpoint:
        raise TrajectoryError(f"Invalid checkpoint name: {checkpoint!r}")
    episodes = list(dict.fromkeys(args.episodes))
    if not episodes:
        raise TrajectoryError("At least one episode must be selected")
    invalid = [episode for episode in episodes if episode < 0 or episode >= evalmod.EXPECTED_EPISODES]
    if invalid:
        raise TrajectoryError(f"Episode does not exist; valid range is [0, 9], got {invalid}")
    return checkpoint, episodes


def reconstruct_episode(
    checkpoint: str,
    episode_index: int,
    dataset: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    evalmod: Any,
    torch: Any,
    np: Any,
) -> tuple[Any, Any]:
    start, end = evalmod.episode_bounds(dataset, episode_index)
    predicted_chunks: list[Any] = []
    ground_truth_chunks: list[Any] = []

    for absolute_start in range(start, end, evalmod.EXPECTED_CHUNK_SIZE):
        relative_start = absolute_start - start
        try:
            item = dataset[absolute_start]
        except Exception as exc:
            raise TrajectoryError(
                f"Dataset decoding failed at episode={episode_index}, frame={relative_start}: {exc}"
            ) from exc
        if evalmod.scalar_int(item.get("episode_index")) != episode_index:
            raise TrajectoryError(
                f"Dataset frame belongs to wrong episode at absolute index {absolute_start}"
            )
        if evalmod.scalar_int(item.get("frame_index")) != relative_start:
            raise TrajectoryError(
                f"Unexpected frame_index at episode={episode_index}: "
                f"{evalmod.scalar_int(item.get('frame_index'))} != {relative_start}"
            )

        observation: dict[str, Any] = {}
        for key in ("observation.state", *evalmod.EXPECTED_CAMERAS):
            if key not in item:
                raise TrajectoryError(f"Dataset item is missing required feature: {key}")
            evalmod.require_finite_tensor(f"Dataset {key}", item[key])
            observation[key] = item[key]

        target = item.get("action")
        padding = item.get("action_is_pad")
        evalmod.require_finite_tensor("Dataset action chunk", target)
        if tuple(target.shape) != (evalmod.EXPECTED_CHUNK_SIZE, evalmod.EXPECTED_ACTION_DIM):
            raise TrajectoryError(f"Ground-truth action chunk has invalid shape: {tuple(target.shape)}")
        if not isinstance(padding, torch.Tensor) or padding.dtype != torch.bool:
            raise TrajectoryError("action_is_pad is missing or is not a bool tensor")
        if tuple(padding.shape) != (evalmod.EXPECTED_CHUNK_SIZE,):
            raise TrajectoryError(f"action_is_pad has invalid shape: {tuple(padding.shape)}")

        try:
            processed_observation = preprocessor(observation)
            with torch.inference_mode():
                prediction = policy.predict_action_chunk(processed_observation)
            evalmod.require_finite_tensor("Normalized prediction", prediction)
            prediction = postprocessor(prediction)
        except Exception as exc:
            raise TrajectoryError(
                f"Inference failed for checkpoint={checkpoint}, episode={episode_index}, "
                f"frame={relative_start}: {exc}"
            ) from exc
        evalmod.require_finite_tensor("Postprocessed prediction", prediction)
        if tuple(prediction.shape) != (
            1,
            evalmod.EXPECTED_CHUNK_SIZE,
            evalmod.EXPECTED_ACTION_DIM,
        ):
            raise TrajectoryError(f"Predicted action chunk has invalid shape: {tuple(prediction.shape)}")

        prediction = prediction.squeeze(0).detach().cpu()
        target = target.detach().cpu()
        if tuple(prediction.shape) != tuple(target.shape):
            raise TrajectoryError(
                f"Prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        valid_mask = ~padding.detach().cpu()
        expected_valid = min(evalmod.EXPECTED_CHUNK_SIZE, end - absolute_start)
        valid_count = int(valid_mask.sum().item())
        if valid_count != expected_valid:
            raise TrajectoryError(
                f"Padding mask valid count mismatch at episode={episode_index}, frame={relative_start}: "
                f"{valid_count} != {expected_valid}"
            )
        predicted_chunks.append(prediction[valid_mask].numpy())
        ground_truth_chunks.append(target[valid_mask].numpy())

    predicted_action = np.concatenate(predicted_chunks, axis=0).astype(np.float32, copy=False)
    ground_truth_action = np.concatenate(ground_truth_chunks, axis=0).astype(np.float32, copy=False)
    expected_length = end - start
    expected_shape = (expected_length, evalmod.EXPECTED_ACTION_DIM)
    if predicted_action.shape != expected_shape or ground_truth_action.shape != expected_shape:
        raise TrajectoryError(
            f"Reconstructed trajectory length mismatch for episode={episode_index}: "
            f"prediction={predicted_action.shape}, ground_truth={ground_truth_action.shape}, "
            f"expected={expected_shape}"
        )
    if not np.isfinite(predicted_action).all() or not np.isfinite(ground_truth_action).all():
        raise TrajectoryError(f"Reconstructed episode {episode_index} contains NaN/Inf")
    return ground_truth_action, predicted_action


ERROR_FIELDS = (
    "frame_index",
    "joint_1_abs_error_deg",
    "joint_2_abs_error_deg",
    "joint_3_abs_error_deg",
    "joint_4_abs_error_deg",
    "joint_5_abs_error_deg",
    "joint_6_abs_error_deg",
    "arm_mean_abs_error_deg",
    "gripper_abs_error_pct",
)


def write_errors_csv(path: Path, ground_truth: Any, prediction: Any, np: Any) -> tuple[Any, Any]:
    absolute_error = np.abs(prediction - ground_truth)
    arm_error_deg = np.rad2deg(absolute_error[:, :6])
    gripper_error = absolute_error[:, 6]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(ERROR_FIELDS)
        for frame_index in range(ground_truth.shape[0]):
            joint_errors = arm_error_deg[frame_index]
            writer.writerow(
                [
                    frame_index,
                    *(f"{float(value):.10f}" for value in joint_errors),
                    f"{float(joint_errors.mean()):.10f}",
                    f"{float(gripper_error[frame_index]):.10f}",
                ]
            )
    return arm_error_deg, gripper_error


def render_episode(
    plot_python: Path,
    npz_path: Path,
    output_dir: Path,
    checkpoint: str,
    episode_index: int,
) -> None:
    command = [
        str(plot_python),
        str(Path(__file__).resolve()),
        "--render-npz",
        str(npz_path),
        "--render-output-dir",
        str(output_dir),
        "--render-checkpoint",
        checkpoint,
        "--render-episode",
        str(episode_index),
    ]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=plot_probe_environment(),
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "no error output"
        raise TrajectoryError(
            f"matplotlib rendering failed for episode {episode_index} with {plot_python}: {details}"
        )


def make_episode_metrics(
    episode_index: int,
    length: int,
    arm_error_deg: Any,
    gripper_error: Any,
    np: Any,
) -> dict[str, Any]:
    joint_flat_index = int(np.argmax(arm_error_deg))
    joint_max_frame, joint_max_index = np.unravel_index(joint_flat_index, arm_error_deg.shape)
    gripper_max_frame = int(np.argmax(gripper_error))
    return {
        "episode_index": episode_index,
        "length": length,
        "arm_mae_deg": float(arm_error_deg.mean()),
        "arm_p95_deg": float(np.percentile(arm_error_deg, 95)),
        "arm_max_deg": float(arm_error_deg.max()),
        "joint_max_error_frame": int(joint_max_frame),
        "joint_max_error_joint": int(joint_max_index) + 1,
        "gripper_mae_pct": float(gripper_error.mean()),
        "gripper_max_pct": float(gripper_error.max()),
        "gripper_max_error_frame": gripper_max_frame,
    }


def write_report(
    path: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    plot_python: Path,
    metrics: list[dict[str, Any]],
) -> None:
    lines = [
        "TB6-R5 ACT trajectory plots",
        "=" * 31,
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Dataset root: {dataset_root}",
        f"Checkpoint path: {checkpoint_path}",
        f"Matplotlib renderer Python: {plot_python}",
        "Episode lengths: " + ", ".join(
            f"episode {row['episode_index']}={row['length']}" for row in metrics
        ),
        "",
    ]
    for row in metrics:
        lines.extend(
            [
                f"Episode {row['episode_index']:03d}",
                f"  length: {row['length']}",
                f"  arm MAE (deg): {row['arm_mae_deg']:.6f}",
                f"  arm P95 (deg): {row['arm_p95_deg']:.6f}",
                f"  arm max (deg): {row['arm_max_deg']:.6f}",
                f"  joint max-error frame: {row['joint_max_error_frame']} "
                f"(joint {row['joint_max_error_joint']})",
                f"  gripper MAE (pct): {row['gripper_mae_pct']:.6f}",
                f"  gripper max (pct): {row['gripper_max_pct']:.6f}",
                f"  gripper max-error frame: {row['gripper_max_error_frame']}",
                "",
            ]
        )
    lines.extend(
        [
            "Semantics",
            "---------",
            "Official ACT predict_action_chunk() is called at episode-local frames 0, 100, 200, ... .",
            "Each checkpoint-normalized action chunk is restored with that checkpoint's saved",
            "postprocessor, cropped by action_is_pad, and concatenated in chronological order.",
            "The validation Dataset stats.json is not used for policy normalization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> Path:
    import numpy as np
    import torch

    import lerobot

    args = parse_args()
    evalmod = import_evaluation_module()
    checkpoint, episodes = validate_requested(args, evalmod)
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoints_root = args.checkpoints_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    pretrained_model = checkpoints_root / checkpoint / "pretrained_model"

    if lerobot.__version__ != evalmod.EXPECTED_LEROBOT_VERSION:
        raise TrajectoryError(
            f"LeRobot version must be {evalmod.EXPECTED_LEROBOT_VERSION}, got {lerobot.__version__}"
        )
    evalmod.validate_dataset_metadata(dataset_root)
    evalmod.validate_checkpoint(pretrained_model)
    dataset = evalmod.make_dataset(dataset_root, args.video_backend)
    episode_lengths: dict[int, int] = {}
    for episode_index in episodes:
        start, end = evalmod.episode_bounds(dataset, episode_index)
        episode_lengths[episode_index] = end - start
    plot_python = find_plot_python(args.plot_python)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== TB6-R5 ACT trajectory visualization ===")
    print(f"LeRobot version : {lerobot.__version__}")
    print(f"torch version   : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"GPU name        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Device          : {device}")
    print(f"Dataset root    : {dataset_root}")
    print(f"Checkpoint path : {pretrained_model}")
    print(f"Selected episodes: {episodes}")
    for episode_index in episodes:
        print(f"Episode {episode_index:03d} length: {episode_lengths[episode_index]}")
    print(f"Plot renderer   : {plot_python}")
    print("Safety          : offline inference only; no robot/control interfaces")
    print()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / stamp
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        raise TrajectoryError(f"Failed to create output directory {output_dir}: {exc}") from exc
    print(f"Output directory: {output_dir}")

    policy, preprocessor, postprocessor, runtime_config = evalmod.load_policy_and_processors(
        pretrained_model, device
    )
    if (
        runtime_config.chunk_size != evalmod.EXPECTED_CHUNK_SIZE
        or runtime_config.n_action_steps != evalmod.EXPECTED_N_ACTION_STEPS
        or runtime_config.n_obs_steps != evalmod.EXPECTED_N_OBS_STEPS
        or runtime_config.temporal_ensemble_coeff is not None
    ):
        raise TrajectoryError(f"Unexpected runtime ACT config for checkpoint {checkpoint}")

    metrics: list[dict[str, Any]] = []
    for episode_index in episodes:
        print(f"[{checkpoint}] Reconstructing episode {episode_index} ...", flush=True)
        ground_truth, prediction = reconstruct_episode(
            checkpoint,
            episode_index,
            dataset,
            policy,
            preprocessor,
            postprocessor,
            evalmod,
            torch,
            np,
        )
        expected_length = episode_lengths[episode_index]
        if ground_truth.shape != (expected_length, 7) or prediction.shape != (expected_length, 7):
            raise TrajectoryError(
                f"Episode {episode_index} prediction length and ground-truth length differ"
            )

        episode_dir = output_dir / f"episode_{episode_index:03d}"
        episode_dir.mkdir(parents=False, exist_ok=False)
        npz_path = episode_dir / f"episode_{episode_index:03d}_actions.npz"
        np.savez_compressed(
            npz_path,
            ground_truth_action=ground_truth,
            predicted_action=prediction,
        )
        errors_path = episode_dir / f"episode_{episode_index:03d}_errors.csv"
        arm_error_deg, gripper_error = write_errors_csv(
            errors_path, ground_truth, prediction, np
        )
        render_episode(
            plot_python,
            npz_path,
            episode_dir,
            checkpoint,
            episode_index,
        )
        row = make_episode_metrics(
            episode_index,
            expected_length,
            arm_error_deg,
            gripper_error,
            np,
        )
        metrics.append(row)
        print(
            f"[{checkpoint}] episode={episode_index} length={expected_length} "
            f"arm_mae_deg={row['arm_mae_deg']:.6f} "
            f"gripper_mae_pct={row['gripper_mae_pct']:.6f}",
            flush=True,
        )

    report_path = output_dir / "report_plots.txt"
    write_report(report_path, dataset_root, pretrained_model, plot_python, metrics)
    print(f"\n[PASS] Trajectory visualization completed: {output_dir}")
    for episode_index in episodes:
        print(f"[OUTPUT] {output_dir / f'episode_{episode_index:03d}'}")
    print(f"[OUTPUT] {report_path}")
    return output_dir


def main() -> int:
    try:
        run()
    except KeyboardInterrupt:
        print("[FAIL] Trajectory visualization interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--render-npz":
        try:
            raise SystemExit(render_npz(sys.argv[1:]))
        except Exception as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            raise SystemExit(1)
    raise SystemExit(main())
