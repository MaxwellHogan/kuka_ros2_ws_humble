#!/usr/bin/env python3
import glob
import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------
# Quaternion helpers
# --------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    # q = [x, y, z, w]
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Hamilton product, q=[x,y,z,w]
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w], dtype=float)


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Smallest rotation angle between orientations q1 and q2 in degrees.
    Uses relative quaternion q_rel = q1^-1 * q2 and angle = 2*acos(|w|).
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    qrel = quat_multiply(quat_conjugate(q1), q2)
    qrel = quat_normalize(qrel)
    w = float(np.clip(abs(qrel[3]), -1.0, 1.0))  # abs handles quaternion double cover
    angle_rad = 2.0 * math.acos(w)
    return math.degrees(angle_rad)


def summarize(name: str, arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{name}_{k}": float("nan") for k in ["n","mean","median","std","rmse","max"]}
    rmse = math.sqrt(float(np.mean(arr**2)))
    return {
        f"{name}_n": int(arr.size),
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{name}_rmse": float(rmse),
        f"{name}_max": float(np.max(arr)),
    }


# --------------------------
# Column mapping
# --------------------------
@dataclass
class Columns:
    # robot end pose (recommended for accuracy checking)
    r_end_px: str
    r_end_py: str
    r_end_pz: str
    r_end_qx: str
    r_end_qy: str
    r_end_qz: str
    r_end_qw: str

    # optitrack end pose (ground truth)
    o_end_px: str
    o_end_py: str
    o_end_pz: str
    o_end_qx: str
    o_end_qy: str
    o_end_qz: str
    o_end_qw: str


def default_columns() -> Columns:
    # Matches your original header names (end pose fields)
    return Columns(
        r_end_px="robot_tool_end_px",
        r_end_py="robot_tool_end_py",
        r_end_pz="robot_tool_end_pz",
        r_end_qx="robot_tool_end_qx",
        r_end_qy="robot_tool_end_qy",
        r_end_qz="robot_tool_end_qz",
        r_end_qw="robot_tool_end_qw",
        o_end_px="optitrack_end_px",
        o_end_py="optitrack_end_py",
        o_end_pz="optitrack_end_pz",
        o_end_qx="optitrack_end_qx",
        o_end_qy="optitrack_end_qy",
        o_end_qz="optitrack_end_qz",
        o_end_qw="optitrack_end_qw",
    )


def read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__file__"] = os.path.basename(path)
    df["__row__"] = np.arange(len(df), dtype=int)  # step index within run
    return df


def compute_errors(df: pd.DataFrame, cols: Columns) -> pd.DataFrame:
    # Filter out rows where optitrack is missing (you used zeros as sentinel)
    o_pos = df[[cols.o_end_px, cols.o_end_py, cols.o_end_pz]].to_numpy(dtype=float)
    valid_opti = ~(np.all(np.isclose(o_pos, 0.0), axis=1))

    r_pos = df[[cols.r_end_px, cols.r_end_py, cols.r_end_pz]].to_numpy(dtype=float)
    pos_err = np.linalg.norm(r_pos - o_pos, axis=1)

    # Rotation error
    rot_err = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        if not valid_opti[i]:
            continue
        rq = df.loc[df.index[i], [cols.r_end_qx, cols.r_end_qy, cols.r_end_qz, cols.r_end_qw]].to_numpy(dtype=float)
        oq = df.loc[df.index[i], [cols.o_end_qx, cols.o_end_qy, cols.o_end_qz, cols.o_end_qw]].to_numpy(dtype=float)
        rot_err[i] = quat_angle_deg(rq, oq)

    out = df.copy()
    out["pos_err_m"] = pos_err
    out["rot_err_deg"] = rot_err
    out["valid_opti"] = valid_opti
    return out


def compute_robot_repeatability(all_df: pd.DataFrame, cols: Columns) -> pd.DataFrame:
    """
    Repeatability across runs:
    For each step index (__row__), measure spread of robot end pose across files.
    """
    # For each step, collect positions and orientations across runs
    rep_rows = []
    for step, g in all_df.groupby("__row__"):
        if len(g) < 2:
            continue

        P = g[[cols.r_end_px, cols.r_end_py, cols.r_end_pz]].to_numpy(dtype=float)

        # Position repeatability: RMS radius around mean + component std
        P_mean = np.mean(P, axis=0)
        dP = P - P_mean
        radii = np.linalg.norm(dP, axis=1)
        pos_rms = math.sqrt(float(np.mean(radii**2)))
        pos_std = float(np.std(radii, ddof=1)) if len(radii) > 1 else 0.0

        # Rotation repeatability:
        # Use mean quaternion approximation by picking first as reference then measuring angles to it.
        # (More robust averaging possible, but this is a good practical metric.)
        q_ref = g.iloc[0][[cols.r_end_qx, cols.r_end_qy, cols.r_end_qz, cols.r_end_qw]].to_numpy(dtype=float)
        q_ref = quat_normalize(q_ref)

        angs = []
        for _, row in g.iterrows():
            q = row[[cols.r_end_qx, cols.r_end_qy, cols.r_end_qz, cols.r_end_qw]].to_numpy(dtype=float)
            angs.append(quat_angle_deg(q_ref, q))
        angs = np.asarray(angs, dtype=float)
        rot_rms = math.sqrt(float(np.mean(angs**2)))
        rot_std = float(np.std(angs, ddof=1)) if len(angs) > 1 else 0.0

        rep_rows.append({
            "step_index": int(step),
            "n_runs": int(len(g)),
            "pos_repeat_rms_m": pos_rms,
            "pos_repeat_std_m": pos_std,
            "rot_repeat_rms_deg": rot_rms,
            "rot_repeat_std_deg": rot_std,
        })

    return pd.DataFrame(rep_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="benchmark_tests", help="Directory containing csv logs")
    parser.add_argument("--pattern", default="kr50_benchmark_log_*.csv", help="Filename glob pattern")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files found under {args.dir}/{args.pattern}")

    cols = default_columns()

    dfs = [read_one_csv(p) for p in paths]
    all_df = pd.concat(dfs, ignore_index=True)

    # Compute errors per row
    all_df = compute_errors(all_df, cols)

    # Overall accuracy stats (robot vs opti)
    valid = all_df[all_df["valid_opti"]].copy()
    pos_stats = summarize("pos_err_m", valid["pos_err_m"].to_numpy())
    rot_stats = summarize("rot_err_deg", valid["rot_err_deg"].to_numpy())

    print("\n=== Overall robot vs OptiTrack error (END pose) ===")
    for k, v in {**pos_stats, **rot_stats}.items():
        print(f"{k}: {v}")

    # Per-run summary (optional but helpful)
    print("\n=== Per-run summary (mean pos/rot error) ===")
    per_run = valid.groupby("__file__").agg(
        n=("pos_err_m", "size"),
        pos_mean=("pos_err_m", "mean"),
        pos_rmse=("pos_err_m", lambda x: math.sqrt(float(np.mean(np.asarray(x)**2)))),
        pos_max=("pos_err_m", "max"),
        rot_mean=("rot_err_deg", "mean"),
        rot_rmse=("rot_err_deg", lambda x: math.sqrt(float(np.mean(np.asarray(x)**2)))),
        rot_max=("rot_err_deg", "max"),
    ).reset_index()
    print(per_run.to_string(index=False))

    # Repeatability (robot end pose across runs)
    rep = compute_robot_repeatability(all_df, cols)
    if rep.empty:
        print("\nRepeatability: not enough runs/steps to compute (need >=2 runs).")
        return

    print("\n=== Robot repeatability across runs (END pose, grouped by step index) ===")
    print(rep.to_string(index=False))

    # Overall repeatability summary across all steps
    rep_pos = summarize("pos_repeat_rms_m", rep["pos_repeat_rms_m"].to_numpy())
    rep_rot = summarize("rot_repeat_rms_deg", rep["rot_repeat_rms_deg"].to_numpy())
    print("\n=== Overall repeatability summary (across step indices) ===")
    for k, v in {**rep_pos, **rep_rot}.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
