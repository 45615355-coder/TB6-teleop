#!/usr/bin/env python3
"""Offline per-session wrist-rotation baseline analysis for MuJoCo JSONL logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


ACTIVE_START = "HOLD -> ACTIVE"
ACTIVE_END_PREFIX = "ACTIVE -> HOLD"
MODES = ("yaw", "pitch", "roll")
CSV_COLUMNS = (
    "time_s", "session_id", "mode", "xr_dx_mm", "xr_dy_mm", "xr_dz_mm",
    "xr_drift_3d_mm", "corrected_dx_mm", "corrected_dy_mm", "corrected_dz_mm",
    "corrected_drift_3d_mm", "tcp_dx_mm", "tcp_dy_mm", "tcp_dz_mm",
    "tcp_drift_3d_mm", "relative_rotation_deg",
)


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _rotation3(value: Any) -> tuple[tuple[float, float, float], ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    rows = tuple(_vector3(row) for row in value)
    return rows if all(row is not None for row in rows) else None


def _sample(record: dict[str, Any]) -> dict[str, Any] | None:
    position = _vector3(record.get("xr_position"))
    corrected_position = _vector3(record.get("xr_corrected_position"))
    tcp_position = _vector3(record.get("mujoco_tcp_position"))
    rotation = _rotation3(record.get("xr_rotation"))
    try:
        timestamp = float(record["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or position is None or corrected_position is None or tcp_position is None or rotation is None:
        return None
    return {"timestamp": timestamp, "xr_position": position, "corrected_position": corrected_position,
            "tcp_position": tcp_position, "xr_rotation": rotation}


def read_active_sessions(log_path: Path) -> tuple[list[list[dict[str, Any]]], int]:
    """Return valid samples from each ACTIVE interval without joining intervals."""
    sessions: list[list[dict[str, Any]]] = []
    active: list[dict[str, Any]] | None = None
    skipped_active_records = 0
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {log_path}:{line_number}: {exc.msg}") from exc
            transition = str(record.get("transition", ""))
            if transition == ACTIVE_START:
                if active:
                    sessions.append(active)
                active = []
            elif transition.startswith(ACTIVE_END_PREFIX):
                if active:
                    sessions.append(active)
                active = None
                continue
            if active is not None:
                sample = _sample(record)
                if sample is None:
                    skipped_active_records += 1
                else:
                    active.append(sample)
    if active:
        sessions.append(active)
    return sessions, skipped_active_records


def _subtract_mm(value: tuple[float, float, float], origin: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple((current - initial) * 1000.0 for current, initial in zip(value, origin))


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(item * item for item in vector))


def _relative_rotation_deg(current: tuple[tuple[float, float, float], ...], origin: tuple[tuple[float, float, float], ...]) -> float:
    relative_trace = sum(
        current[row][column] * origin[row][column]
        for row in range(3) for column in range(3)
    )
    cosine = max(-1.0, min(1.0, (relative_trace - 1.0) * 0.5))
    return math.degrees(math.acos(cosine))


def _range(values: Iterable[float]) -> float:
    values = list(values)
    return max(values) - min(values)


def analyze_session(session_id: int, samples: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not samples:
        raise ValueError(f"session {session_id} has no valid samples")
    first = samples[0]
    first_time = first["timestamp"]
    first_xr = first["xr_position"]
    first_corrected = first["corrected_position"]
    first_tcp = first["tcp_position"]
    first_rotation = first["xr_rotation"]
    rows: list[dict[str, Any]] = []
    for sample in samples:
        xr_delta = _subtract_mm(sample["xr_position"], first_xr)
        corrected_delta = _subtract_mm(sample["corrected_position"], first_corrected)
        tcp_delta = _subtract_mm(sample["tcp_position"], first_tcp)
        rows.append({
            "time_s": sample["timestamp"] - first_time,
            "session_id": session_id,
            "mode": mode,
            "xr_dx_mm": xr_delta[0], "xr_dy_mm": xr_delta[1], "xr_dz_mm": xr_delta[2],
            "xr_drift_3d_mm": _norm(xr_delta),
            "corrected_dx_mm": corrected_delta[0], "corrected_dy_mm": corrected_delta[1], "corrected_dz_mm": corrected_delta[2],
            "corrected_drift_3d_mm": _norm(corrected_delta),
            "tcp_dx_mm": tcp_delta[0], "tcp_dy_mm": tcp_delta[1], "tcp_dz_mm": tcp_delta[2],
            "tcp_drift_3d_mm": _norm(tcp_delta),
            "relative_rotation_deg": _relative_rotation_deg(sample["xr_rotation"], first_rotation),
        })
    def metrics(prefix: str) -> dict[str, float]:
        return {
            "x_range_mm": _range(row[f"{prefix}_dx_mm"] for row in rows),
            "y_range_mm": _range(row[f"{prefix}_dy_mm"] for row in rows),
            "z_range_mm": _range(row[f"{prefix}_dz_mm"] for row in rows),
            "max_3d_drift_mm": max(row[f"{prefix}_drift_3d_mm"] for row in rows),
            "final_3d_drift_mm": rows[-1][f"{prefix}_drift_3d_mm"],
        }
    return rows, {
        "session_id": session_id,
        "samples": len(rows),
        "duration_s": rows[-1]["time_s"],
        "quest_raw": metrics("xr"),
        "quest_corrected": metrics("corrected"),
        "mujoco_tcp": metrics("tcp"),
        "rotation": {
            "max_relative_rotation_deg": max(row["relative_rotation_deg"] for row in rows),
            "final_relative_rotation_deg": rows[-1]["relative_rotation_deg"],
        },
    }


def _max_metric(summaries: list[dict[str, Any]], group: str, name: str) -> float:
    return max(summary[group][name] for summary in summaries)


def overall_summary(summaries: list[dict[str, Any]], skipped_active_records: int) -> dict[str, Any]:
    if not summaries:
        raise ValueError("no valid ACTIVE sessions found")
    metric_names = ("x_range_mm", "y_range_mm", "z_range_mm", "max_3d_drift_mm", "final_3d_drift_mm")
    return {
        "sessions": len(summaries),
        "total_active_duration_s": sum(summary["duration_s"] for summary in summaries),
        "total_active_samples": sum(summary["samples"] for summary in summaries),
        "skipped_active_records": skipped_active_records,
        "aggregation": "Per-session origins are retained; each aggregate metric is the maximum across sessions, never a concatenated trajectory.",
        "quest_raw_max_across_sessions": {name: _max_metric(summaries, "quest_raw", name) for name in metric_names},
        "quest_corrected_max_across_sessions": {name: _max_metric(summaries, "quest_corrected", name) for name in metric_names},
        "mujoco_tcp_max_across_sessions": {name: _max_metric(summaries, "mujoco_tcp", name) for name in metric_names},
        "rotation_max_across_sessions": {
            "max_relative_rotation_deg": _max_metric(summaries, "rotation", "max_relative_rotation_deg"),
            "final_relative_rotation_deg": _max_metric(summaries, "rotation", "final_relative_rotation_deg"),
        },
    }


def write_outputs(log_path: Path, mode: str, rows: list[dict[str, Any]], session_summaries: list[dict[str, Any]], skipped: int) -> tuple[Path, Path, dict[str, Any]]:
    analysis_directory = Path(__file__).resolve().parent / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{log_path.stem}_{mode}"
    csv_path = analysis_directory / f"{stem}.csv"
    summary_path = analysis_directory / f"{stem}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    overall = overall_summary(session_summaries, skipped)
    payload = {"source_log": str(log_path.resolve()), "mode": mode, "sessions": session_summaries, "overall": overall}
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path, summary_path, overall


def print_summary(mode: str, overall: dict[str, Any], csv_path: Path, summary_path: Path) -> None:
    quest, corrected, tcp, rotation = (overall["quest_raw_max_across_sessions"], overall["quest_corrected_max_across_sessions"], overall["mujoco_tcp_max_across_sessions"], overall["rotation_max_across_sessions"])
    print("========== WRIST ROTATION BASELINE ==========")
    print(f"mode: {mode}")
    print(f"sessions: {overall['sessions']}")
    print(f"duration: {overall['total_active_duration_s']:.1f} s")
    print("\nQuest raw (max across independent sessions):")
    print(f"X range: {quest['x_range_mm']:.1f} mm")
    print(f"Y range: {quest['y_range_mm']:.1f} mm")
    print(f"Z range: {quest['z_range_mm']:.1f} mm")
    print(f"max 3D drift: {quest['max_3d_drift_mm']:.1f} mm")
    print("\nQuest wrist-pivot corrected (max across independent sessions):")
    print(f"X range: {corrected['x_range_mm']:.1f} mm")
    print(f"Y range: {corrected['y_range_mm']:.1f} mm")
    print(f"Z range: {corrected['z_range_mm']:.1f} mm")
    print(f"max 3D drift: {corrected['max_3d_drift_mm']:.1f} mm")
    print("\nMuJoCo TCP (max across independent sessions):")
    print(f"X range: {tcp['x_range_mm']:.1f} mm")
    print(f"Y range: {tcp['y_range_mm']:.1f} mm")
    print(f"Z range: {tcp['z_range_mm']:.1f} mm")
    print(f"max 3D drift: {tcp['max_3d_drift_mm']:.1f} mm")
    print("\nRotation:")
    print(f"max relative rotation: {rotation['max_relative_rotation_deg']:.1f} deg")
    print(f"\nCSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print("=============================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline wrist rotation analysis; mode is a user-supplied trial label.")
    parser.add_argument("log", type=Path, help="Existing MuJoCo JSONL log")
    parser.add_argument("--mode", choices=MODES, required=True, help="Manual trial label; never used to transform or filter data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log.is_file():
        raise SystemExit(f"log file not found: {args.log}")
    sessions, skipped = read_active_sessions(args.log)
    if not sessions:
        raise SystemExit("no valid ACTIVE sessions found; expected transition 'HOLD -> ACTIVE' in the JSONL")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for session_id, session in enumerate(sessions, 1):
        session_rows, summary = analyze_session(session_id, session, args.mode)
        rows.extend(session_rows)
        summaries.append(summary)
    csv_path, summary_path, overall = write_outputs(args.log, args.mode, rows, summaries, skipped)
    print_summary(args.mode, overall, csv_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
