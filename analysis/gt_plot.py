# analysis/gt_plot.py
from __future__ import annotations
import os
import numpy as np
import matplotlib
import csv
from typing import Optional, Dict
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    # Normalize to be safe
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = (q / n)

    # Standard quaternion -> rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)
    return R


def _alpha_about_y_deg_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    """
    alpha = rotation about Y axis (xz-plane in-plane rotation).
    For pure Ry(alpha): R = [[c,0,s],[0,1,0],[-s,0,c]]
    => alpha = atan2(R[0,2], R[2,2])
    """
    R = _quat_to_rotmat(qw, qx, qy, qz)
    alpha = float(np.arctan2(R[0, 2], R[2, 2]))
    return alpha * 180.0 / np.pi


def _unwrap_deg(a: np.ndarray) -> np.ndarray:
    # unwrap in radians then back to deg
    ar = np.deg2rad(a.astype(np.float64))
    ar_u = np.unwrap(ar)
    return np.rad2deg(ar_u)

def load_em_perframe_motion(
    tracker_csv_path: str,
    port: str = "Port:11",
    axis_map: dict | None = None,
) -> dict:
    """
    CSV reader without pandas.
    Returns dict with:
      em_dx_mm, em_dy_mm, em_dz_mm, em_dalpha_deg (each shape n-1)
    """
    if axis_map is None:
        axis_map = {"x": "Tx", "y": "Ty", "z": "Tz"}

    rows = []
    with open(tracker_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # filter port
            if r.get("PortHandle", "") != port:
                continue
            # filter enabled if column exists
            if "TransformStatus" in r and r["TransformStatus"] != "Enabled":
                continue
            rows.append(r)

    if not rows:
        raise ValueError(f"No rows found for {port} in {tracker_csv_path}")

    # sort by Frame# if present
    if "Frame#" in rows[0]:
        rows.sort(key=lambda rr: int(float(rr.get("Frame#", "0"))))

    def col(name: str) -> np.ndarray:
        return np.array([float(rr.get(name, "nan")) for rr in rows], dtype=np.float64)

    tx = col(axis_map["x"])
    ty = col(axis_map["y"])
    tz = col(axis_map["z"])

    qw = col("Q0")
    qx = col("Qx")
    qy = col("Qy")
    qz = col("Qz")

    alpha = np.array(
        [_alpha_about_y_deg_from_quat(a, b, c, d) for a, b, c, d in zip(qw, qx, qy, qz)],
        dtype=np.float64,
    )
    alpha = _unwrap_deg(alpha)

    em_dx = np.diff(tx)
    em_dy = np.diff(ty)
    em_dz = np.diff(tz)
    em_dalpha = np.diff(alpha)

    return {
        "em_dx_mm": em_dx,
        "em_dy_mm": em_dy,
        "em_dz_mm": em_dz,
        "em_dalpha_deg": em_dalpha,
    }

def plot_gt_comparison(
    out_png_path: str,
    fh_dx_mm: np.ndarray,
    fh_dy_mm: np.ndarray,
    fh_dz_mm: np.ndarray,
    fh_dalpha_deg: np.ndarray,
    em_dx_mm: np.ndarray,
    em_dy_mm: np.ndarray,
    em_dz_mm: np.ndarray,
    em_dalpha_deg: np.ndarray,
) -> None:
    # align length (use the shortest)
    n = min(len(fh_dx_mm), len(em_dx_mm), len(fh_dy_mm), len(em_dy_mm), len(fh_dz_mm), len(em_dz_mm),
            len(fh_dalpha_deg), len(em_dalpha_deg))
    if n <= 0:
        return

    fh_dx_mm = fh_dx_mm[:n]
    fh_dy_mm = fh_dy_mm[:n]
    fh_dz_mm = fh_dz_mm[:n]
    fh_dalpha_deg = fh_dalpha_deg[:n]
    em_dx_mm = em_dx_mm[:n]
    em_dy_mm = em_dy_mm[:n]
    em_dz_mm = em_dz_mm[:n]
    em_dalpha_deg = em_dalpha_deg[:n]

    x = np.arange(n)

    # errors
    ex = fh_dx_mm - em_dx_mm
    ey = fh_dy_mm - em_dy_mm
    ez = fh_dz_mm - em_dz_mm
    ea = fh_dalpha_deg - em_dalpha_deg

    fig = plt.figure(figsize=(14, 10), dpi=140)

    # --- Row 1: overlay ---
    ax1 = fig.add_subplot(3, 4, 1); ax2 = fig.add_subplot(3, 4, 2)
    ax3 = fig.add_subplot(3, 4, 3); ax4 = fig.add_subplot(3, 4, 4)

    ax1.plot(x, em_dx_mm, label="EM (GT)"); ax1.plot(x, fh_dx_mm, label="Freehand")
    ax1.set_title("Δx per frame (mm)"); ax1.set_xlabel("step i (i→i+1)"); ax1.grid(True, alpha=0.25)

    ax2.plot(x, em_dz_mm, label="EM (GT)"); ax2.plot(x, fh_dz_mm, label="Freehand")
    ax2.set_title("Δz per frame (mm)"); ax2.set_xlabel("step i (i→i+1)"); ax2.grid(True, alpha=0.25)

    ax3.plot(x, em_dy_mm, label="EM (GT)"); ax3.plot(x, fh_dy_mm, label="Freehand")
    ax3.set_title("Δy per frame (mm)"); ax3.set_xlabel("step i (i→i+1)"); ax3.grid(True, alpha=0.25)

    ax4.plot(x, em_dalpha_deg, label="EM (GT)"); ax4.plot(x, fh_dalpha_deg, label="Freehand")
    ax4.set_title("Δalpha about Y per frame (deg)"); ax4.set_xlabel("step i (i→i+1)"); ax4.grid(True, alpha=0.25)

    for ax in (ax1, ax2, ax3, ax4):
        ax.legend(fontsize=8, loc="upper right")

    # --- Row 2: error series ---
    bx1 = fig.add_subplot(3, 4, 5); bx2 = fig.add_subplot(3, 4, 6)
    bx3 = fig.add_subplot(3, 4, 7); bx4 = fig.add_subplot(3, 4, 8)

    bx1.plot(x, ex); bx1.axhline(0, linewidth=1)
    bx1.set_title("Error: Δx (FH - EM) (mm)"); bx1.set_xlabel("step i"); bx1.grid(True, alpha=0.25)

    bx2.plot(x, ez); bx2.axhline(0, linewidth=1)
    bx2.set_title("Error: Δz (FH - EM) (mm)"); bx2.set_xlabel("step i"); bx2.grid(True, alpha=0.25)

    bx3.plot(x, ey); bx3.axhline(0, linewidth=1)
    bx3.set_title("Error: Δy (FH - EM) (mm)"); bx3.set_xlabel("step i"); bx3.grid(True, alpha=0.25)

    bx4.plot(x, ea); bx4.axhline(0, linewidth=1)
    bx4.set_title("Error: Δalpha (FH - EM) (deg)"); bx4.set_xlabel("step i"); bx4.grid(True, alpha=0.25)

    # --- Row 3: boxplot of errors + summary text ---
    cx1 = fig.add_subplot(3, 4, 9)
    cx1.boxplot([ex, ez, ey, ea], labels=["Δx(mm)", "Δz(mm)", "Δy(mm)", "Δα(deg)"])
    cx1.set_title("Error distribution (FH - EM)")
    cx1.grid(True, alpha=0.25)

    # metrics panel
    cx2 = fig.add_subplot(3, 4, 10)
    cx2.axis("off")

    def _rmse(v): return float(np.sqrt(np.mean(v*v))) if len(v) else float("nan")
    def _med(v): return float(np.median(v)) if len(v) else float("nan")
    def _p95(v): return float(np.percentile(np.abs(v), 95)) if len(v) else float("nan")

    lines = [
        f"n steps: {n}",
        "",
        f"Δx RMSE: {_rmse(ex):.4f} mm | median: {_med(ex):+.4f} | |p95|: {_p95(ex):.4f}",
        f"Δz RMSE: {_rmse(ez):.4f} mm | median: {_med(ez):+.4f} | |p95|: {_p95(ez):.4f}",
        f"Δy RMSE: {_rmse(ey):.4f} mm | median: {_med(ey):+.4f} | |p95|: {_p95(ey):.4f}",
        f"Δα RMSE: {_rmse(ea):.4f} deg | median: {_med(ea):+.4f} | |p95|: {_p95(ea):.4f}",
    ]
    cx2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png_path) or ".", exist_ok=True)
    fig.savefig(out_png_path)
    plt.close(fig)


def _stats(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "sum": np.nan,
            "rmse0": np.nan, "mae": np.nan, "p95_abs": np.nan
        }
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "sum": float(np.sum(v)),
        "rmse0": float(np.sqrt(np.mean(v * v))),   # RMSE vs 0 (magnitude)
        "mae": float(np.mean(np.abs(v))),
        "p95_abs": float(np.percentile(np.abs(v), 95)),
    }


def plot_gt_summary(
    out_png_path: str,
    fh_dx_mm: np.ndarray,
    fh_dy_mm: np.ndarray,
    fh_dz_mm: np.ndarray,
    fh_dalpha_deg: np.ndarray,
    em_dx_mm: np.ndarray,
    em_dy_mm: np.ndarray,
    em_dz_mm: np.ndarray,
    em_dalpha_deg: np.ndarray,
) -> None:
    """
    Summary-only GT plot (no per-step time series):
    - Compare overall mean/std/sum (FH vs EM) for x,y,z,alpha
    - Show error distributions (FH - EM) via boxplot + histogram
    - Add text panel with RMSE/MAE/p95 metrics
    """
    fh_dx_mm = np.asarray(fh_dx_mm, dtype=np.float64)
    fh_dy_mm = np.asarray(fh_dy_mm, dtype=np.float64)
    fh_dz_mm = np.asarray(fh_dz_mm, dtype=np.float64)
    fh_da = np.asarray(fh_dalpha_deg, dtype=np.float64)

    em_dx_mm = np.asarray(em_dx_mm, dtype=np.float64)
    em_dy_mm = np.asarray(em_dy_mm, dtype=np.float64)
    em_dz_mm = np.asarray(em_dz_mm, dtype=np.float64)
    em_da = np.asarray(em_dalpha_deg, dtype=np.float64)

    # Align lengths by shortest
    n = min(
        len(fh_dx_mm), len(em_dx_mm),
        len(fh_dy_mm), len(em_dy_mm),
        len(fh_dz_mm), len(em_dz_mm),
        len(fh_da), len(em_da),
    )
    if n <= 0:
        return

    fh_dx_mm = fh_dx_mm[:n]; em_dx_mm = em_dx_mm[:n]
    fh_dy_mm = fh_dy_mm[:n]; em_dy_mm = em_dy_mm[:n]
    fh_dz_mm = fh_dz_mm[:n]; em_dz_mm = em_dz_mm[:n]
    fh_da = fh_da[:n];       em_da = em_da[:n]

    ex = fh_dx_mm - em_dx_mm
    ey = fh_dy_mm - em_dy_mm
    ez = fh_dz_mm - em_dz_mm
    ea = fh_da - em_da

    s_fh_x = _stats(fh_dx_mm); s_em_x = _stats(em_dx_mm); s_e_x = _stats(ex)
    s_fh_y = _stats(fh_dy_mm); s_em_y = _stats(em_dy_mm); s_e_y = _stats(ey)
    s_fh_z = _stats(fh_dz_mm); s_em_z = _stats(em_dz_mm); s_e_z = _stats(ez)
    s_fh_a = _stats(fh_da);    s_em_a = _stats(em_da);    s_e_a = _stats(ea)

    labels = ["Δx (mm)", "Δy (mm)", "Δz (mm)", "Δα (deg)"]
    idx = np.arange(len(labels))
    w = 0.35

    fig = plt.figure(figsize=(18, 10), dpi=160)

    # (1) Mean bars
    ax1 = fig.add_subplot(2, 3, 1)
    fh_means = [s_fh_x["mean"], s_fh_y["mean"], s_fh_z["mean"], s_fh_a["mean"]]
    em_means = [s_em_x["mean"], s_em_y["mean"], s_em_z["mean"], s_em_a["mean"]]
    ax1.bar(idx - w/2, em_means, width=w, label="EM (GT)")
    ax1.bar(idx + w/2, fh_means, width=w, label="Freehand")
    ax1.set_xticks(idx); ax1.set_xticklabels(labels)
    ax1.set_title("Overall mean motion per step")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    # (2) Std bars
    ax2 = fig.add_subplot(2, 3, 2)
    fh_stds = [s_fh_x["std"], s_fh_y["std"], s_fh_z["std"], s_fh_a["std"]]
    em_stds = [s_em_x["std"], s_em_y["std"], s_em_z["std"], s_em_a["std"]]
    ax2.bar(idx - w/2, em_stds, width=w, label="EM (GT)")
    ax2.bar(idx + w/2, fh_stds, width=w, label="Freehand")
    ax2.set_xticks(idx); ax2.set_xticklabels(labels)
    ax2.set_title("Overall std (step-to-step variability)")
    ax2.grid(True, alpha=0.25)

    # (3) Sum bars
    ax3 = fig.add_subplot(2, 3, 3)
    fh_sums = [s_fh_x["sum"], s_fh_y["sum"], s_fh_z["sum"], s_fh_a["sum"]]
    em_sums = [s_em_x["sum"], s_em_y["sum"], s_em_z["sum"], s_em_a["sum"]]
    ax3.bar(idx - w/2, em_sums, width=w, label="EM (GT)")
    ax3.bar(idx + w/2, fh_sums, width=w, label="Freehand")
    ax3.set_xticks(idx); ax3.set_xticklabels(labels)
    ax3.set_title("Total motion (sum of per-step deltas)")
    ax3.grid(True, alpha=0.25)

    # (4) Boxplot errors
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.boxplot([ex, ey, ez, ea], labels=["Δx err(mm)", "Δy err(mm)", "Δz err(mm)", "Δα err(deg)"])
    ax4.axhline(0, linewidth=1)
    ax4.set_title("Error distribution (Freehand - EM)")
    ax4.grid(True, alpha=0.25)

    # (5) Hist errors (overlaid)
    ax5 = fig.add_subplot(2, 3, 5)
    bins = 45
    ax5.hist(ex[np.isfinite(ex)], bins=bins, alpha=0.55, label="Δx err (mm)")
    ax5.hist(ey[np.isfinite(ey)], bins=bins, alpha=0.55, label="Δy err (mm)")
    ax5.hist(ez[np.isfinite(ez)], bins=bins, alpha=0.55, label="Δz err (mm)")
    ax5.hist(ea[np.isfinite(ea)], bins=bins, alpha=0.55, label="Δα err (deg)")
    ax5.axvline(0, linewidth=1)
    ax5.set_title("Error histograms (overlaid)")
    ax5.grid(True, alpha=0.25)
    ax5.legend(fontsize=9)

    # (6) Text metrics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    def _lines(name: str, sfh: dict, sem: dict, se: dict, unit: str) -> list[str]:
        return [
            f"[{name}] unit={unit}, n={se['n']}",
            f"  mean:   FH {sfh['mean']:+.4f} | EM {sem['mean']:+.4f} | err_mean {(sfh['mean']-sem['mean']):+.4f}",
            f"  median: FH {sfh['median']:+.4f} | EM {sem['median']:+.4f}",
            f"  std:    FH {sfh['std']:.4f} | EM {sem['std']:.4f}",
            f"  sum:    FH {sfh['sum']:+.4f} | EM {sem['sum']:+.4f}",
            f"  err_RMSE: {se['rmse0']:.4f} | err_MAE: {se['mae']:.4f} | |err| p95: {se['p95_abs']:.4f}",
            "",
        ]

    text = []
    text += _lines("Δx", s_fh_x, s_em_x, s_e_x, "mm")
    text += _lines("Δy", s_fh_y, s_em_y, s_e_y, "mm")
    text += _lines("Δz", s_fh_z, s_em_z, s_e_z, "mm")
    text += _lines("Δα", s_fh_a, s_em_a, s_e_a, "deg")

    ax6.text(0.0, 1.0, "\n".join(text), va="top", ha="left", fontsize=10)

    fig.suptitle("GT Summary: overall average motion vs EM (x, y, z, alpha)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(out_png_path) or ".", exist_ok=True)
    fig.savefig(out_png_path)
    plt.close(fig)
