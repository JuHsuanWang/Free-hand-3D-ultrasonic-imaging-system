# algorithms/stabilizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import os
import numpy as np
import cv2

from .geometry import generate_grid_points


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA/gray to uint8 gray."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)
    img -= img.min()
    mx = img.max()
    if mx > 0:
        img /= mx
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def track_one_point_ncc(ref: np.ndarray, mov: np.ndarray, cx: int, cz: int, win: int = 16, search: int = 4):
    """NCC tracking for one point; returns (dx, dz, best_cc)."""
    h, w = ref.shape[:2]
    hw = win // 2

    x1, x2 = cx - hw, cx + hw
    z1, z2 = cz - hw, cz + hw
    if x1 < 0 or z1 < 0 or x2 >= w or z2 >= h:
        return 0, 0, -1.0

    template = ref[z1:z2 + 1, x1:x2 + 1]

    sx1 = max(0, x1 - search)
    sz1 = max(0, z1 - search)
    sx2 = min(w - 1, x2 + search)
    sz2 = min(h - 1, z2 + search)

    search_img = mov[sz1:sz2 + 1, sx1:sx2 + 1]
    if search_img.shape[0] < template.shape[0] or search_img.shape[1] < template.shape[1]:
        return 0, 0, -1.0

    res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    _, best_cc, _, best_loc = cv2.minMaxLoc(res)

    best_x = sx1 + best_loc[0] + hw
    best_z = sz1 + best_loc[1] + hw
    dx = best_x - cx
    dz = best_z - cz
    return int(dx), int(dz), float(best_cc)


def solve_rigid_transform_kabsch(P_px: np.ndarray, Q_px: np.ndarray, dx_mm: float = 1.0, dz_mm: float = 1.0):
    """
    Kabsch in mm-space (anisotropic pixels).
    Return theta(rad), tx(px), tz(px) where (tx,tz) is forward transform (ref->mov).
    """
    P = np.column_stack([P_px[:, 0] * dx_mm, P_px[:, 1] * dz_mm]).astype(np.float64)
    Q = np.column_stack([Q_px[:, 0] * dx_mm, Q_px[:, 1] * dz_mm]).astype(np.float64)

    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)

    P0 = P - cP
    Q0 = Q - cQ

    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    theta = np.arctan2(R[1, 0], R[0, 0])

    # Q ≈ R P + t  => t = cQ - R cP
    t_mm = cQ - (R @ cP)

    tx_px = t_mm[0] / dx_mm if dx_mm != 0 else 0.0
    tz_px = t_mm[1] / dz_mm if dz_mm != 0 else 0.0
    return theta, tx_px, tz_px


def warp_back_to_ref(mov: np.ndarray, theta_rad: float, tx_px: float, tz_px: float,
                     border_mode=cv2.BORDER_CONSTANT):
    """Apply inverse transform so that mov aligns back to ref coordinate system."""
    h, w = mov.shape[:2]
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))

    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float64)
    Rt = R.T
    t = np.array([tx_px, tz_px], dtype=np.float64)
    t_inv = -Rt @ t

    M = np.array([[Rt[0, 0], Rt[0, 1], t_inv[0]],
                  [Rt[1, 0], Rt[1, 1], t_inv[1]]], dtype=np.float32)

    channels = mov.shape[2] if mov.ndim > 2 else 1
    if channels == 4:
        border_val = (0, 0, 0, 0)  # transparent
    else:
        border_val = (0, 0, 0)

    return cv2.warpAffine(
        mov, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_val
    )


def _cc_to_color_bgr(cc: float) -> Tuple[int, int, int]:
    """
    Match third code: cc in [-1,1] -> BGR (red low, green high).
    """
    cc = max(-1.0, min(1.0, float(cc)))
    t = (cc + 1.0) * 0.5  # 0..1
    r = int(255 * (1.0 - t))
    g = int(255 * t)
    return (0, g, r)  # BGR


def _make_dbg_image_with_footer(
    base_bgr: np.ndarray,
    per_point: List[Tuple[int, int, int, int, float, bool]],
    mean_cc: float,
    valid_n: int,
    theta_deg: float,
    tx: float,
    tz: float,
    total_points: int = 25,
) -> np.ndarray:
    """
    EXACTLY match third code style:
    - draw valid points as small circles, invalid as tilted cross (red)
    - draw arrowedLine for valid points
    - append white footer with 5 lines:
        Avg_cc, Valid, alpha, tx, tz
    """
    base = base_bgr.copy()

    # draw points + vectors (no per-point cc text)
    for (cx, cz, dx, dz, cc, is_valid) in per_point:
        color = _cc_to_color_bgr(cc)
        if is_valid:
            cv2.circle(base, (cx, cz), 2, color, -1)
        else:
            cv2.drawMarker(
                base, (cx, cz), (0, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1
            )
        if is_valid:
            end = (int(cx + dx), int(cz + dz))
            cv2.arrowedLine(base, (cx, cz), end, color, 1, tipLength=0.25)

    lines = [
        f"Avg_cc {mean_cc:.3f}",
        f"Valid {valid_n:02d}/{total_points}",
        f"alpha {theta_deg:+.2f} deg",
        f"tx {tx:+.2f} px",
        f"tz {tz:+.2f} px",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thick = 1
    line_type = cv2.LINE_AA

    sizes = [cv2.getTextSize(s, font, scale, thick)[0] for s in lines]
    max_h = max(sz[1] for sz in sizes) if sizes else 10
    line_step = max_h + 6
    footer_h = 6 + len(lines) * line_step + 6

    h, w = base.shape[:2]
    footer = np.full((footer_h, w, 3), 255, dtype=np.uint8)

    x = 6
    y = 6 + max_h
    for s in lines:
        cv2.putText(footer, s, (x, y), font, scale, (0, 0, 0), thick, line_type)
        y += line_step

    dbg = np.vstack([base, footer])
    return dbg


@dataclass
class StabilizerConfig:
    """Parameters for NCC+Kabsch stabilization."""
    crop_size: int
    stab_grid: int
    stab_win: int
    stab_search: int
    stab_cc_thresh: float
    dx_mm: float
    dz_mm: float

    # --- NEW: debug output (match third code) ---
    save_debug: bool = False
    debug_out_dir: Optional[str] = None   # e.g., ".../stabilized_frames/R"
    debug_prefix: str = "R"              # file prefix, default "R"
    # ------------------------------------------


class SequenceStabilizer:
    """Stabilize full ROI frames using NCC on the red-box crop region and Kabsch rigid fit."""

    def __init__(self, cfg: StabilizerConfig):
        self.cfg = cfg

    def stabilize_full_roi_inplace(
        self,
        right_frames: np.ndarray,
        left_frames: np.ndarray,
        click_point_xy: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert BGR -> BGRA (adds alpha), then stabilize frames in-place.
        Returns (right_frames_bgra, left_frames_bgra).
        Also optionally saves debug images R_XXXX_dbg.png with EXACT third-code style.
        """
        n = len(right_frames)
        if n < 2:
            return right_frames, left_frames

        # Add alpha channel for transparent padding
        h, w = right_frames[0].shape[:2]
        alpha = np.full((n, h, w, 1), 255, dtype=np.uint8)
        if right_frames.shape[-1] == 3:
            right_frames = np.concatenate([right_frames, alpha], axis=3)
        if left_frames.shape[-1] == 3:
            left_frames = np.concatenate([left_frames, alpha], axis=3)

        # Debug dir
        if self.cfg.save_debug and self.cfg.debug_out_dir:
            os.makedirs(self.cfg.debug_out_dir, exist_ok=True)

        ref_full_gray = to_gray_u8(right_frames[0])

        cx, cy = click_point_xy
        half = self.cfg.crop_size // 2
        crop_x1 = max(0, cx - half)
        crop_y1 = max(0, cy - half)

        ref_crop_gray = ref_full_gray[crop_y1:crop_y1 + self.cfg.crop_size,
                                      crop_x1:crop_x1 + self.cfg.crop_size]

        h_crop, w_crop = ref_crop_gray.shape[:2]
        margin = max(10, self.cfg.stab_win // 2 + self.cfg.stab_search + 2)
        local_pts = generate_grid_points(h_crop, w_crop, grid=self.cfg.stab_grid, margin=margin)

        total_points = int(self.cfg.stab_grid) * int(self.cfg.stab_grid)

        for i in range(1, n):
            mov_full = right_frames[i]
            mov_full_gray = to_gray_u8(mov_full)
            mov_crop_gray = mov_full_gray[crop_y1:crop_y1 + self.cfg.crop_size,
                                          crop_x1:crop_x1 + self.cfg.crop_size]

            P_local, Q_local = [], []
            per_point = []  # (cx,cz, dx,dz, cc, is_valid) in LOCAL crop coords

            for (lx, lz) in local_pts:
                dx, dz, cc = track_one_point_ncc(
                    ref_crop_gray, mov_crop_gray, int(lx), int(lz),
                    win=self.cfg.stab_win, search=self.cfg.stab_search
                )
                is_valid = (cc >= self.cfg.stab_cc_thresh)
                per_point.append((int(lx), int(lz), int(dx), int(dz), float(cc), bool(is_valid)))

                if is_valid:
                    P_local.append([lx, lz])
                    Q_local.append([lx + dx, lz + dz])

            # Not enough correspondences -> skip stabilization for this frame
            if len(P_local) < 6:
                # still allow debug output (theta/tx/tz = 0), matching third-code behavior
                if self.cfg.save_debug and self.cfg.debug_out_dir:
                    # Use original mov frame (converted to BGR for drawing)
                    if mov_full.shape[2] == 4:
                        base_bgr = cv2.cvtColor(mov_full, cv2.COLOR_BGRA2BGR)
                    else:
                        base_bgr = mov_full.copy()

                    # Draw in LOCAL crop coordinates on a CROP view like third code (third debug is on cropped R png)
                    # Here we mimic by drawing on the crop region itself:
                    crop_bgr = base_bgr[crop_y1:crop_y1 + self.cfg.crop_size,
                                        crop_x1:crop_x1 + self.cfg.crop_size].copy()

                    ccs_valid = [pp[4] for pp in per_point if pp[5]]
                    mean_cc = float(np.mean(ccs_valid)) if ccs_valid else -1.0
                    dbg = _make_dbg_image_with_footer(
                        crop_bgr, per_point,
                        mean_cc=mean_cc,
                        valid_n=len(P_local),
                        theta_deg=0.0,
                        tx=0.0,
                        tz=0.0,
                        total_points=total_points,
                    )
                    cv2.imwrite(
                        os.path.join(self.cfg.debug_out_dir, f"{self.cfg.debug_prefix}_{i:04d}_dbg.png"),
                        dbg
                    )

                ref_full_gray = mov_full_gray
                ref_crop_gray = ref_full_gray[crop_y1:crop_y1 + self.cfg.crop_size,
                                              crop_x1:crop_x1 + self.cfg.crop_size]
                continue

            offset = np.array([crop_x1, crop_y1], dtype=np.float64)
            P_global = np.array(P_local, dtype=np.float64) + offset
            Q_global = np.array(Q_local, dtype=np.float64) + offset

            theta, tx, tz = solve_rigid_transform_kabsch(
                P_global, Q_global, dx_mm=self.cfg.dx_mm, dz_mm=self.cfg.dz_mm
            )

            right_frames[i] = warp_back_to_ref(right_frames[i], theta, tx, tz)
            left_frames[i] = warp_back_to_ref(left_frames[i], theta, tx, tz)

            # --- NEW: debug image output (EXACT third-code style) ---
            if self.cfg.save_debug and self.cfg.debug_out_dir:
                aligned_full = right_frames[i]
                if aligned_full.shape[2] == 4:
                    base_bgr = cv2.cvtColor(aligned_full, cv2.COLOR_BGRA2BGR)
                else:
                    base_bgr = aligned_full.copy()

                # Draw on the SAME region style as third: the tracked crop (100x100) region
                crop_bgr = base_bgr[crop_y1:crop_y1 + self.cfg.crop_size,
                                    crop_x1:crop_x1 + self.cfg.crop_size].copy()

                # mean cc uses only valid points (same spirit as third code)
                ccs_valid = [pp[4] for pp in per_point if pp[5]]
                mean_cc = float(np.mean(ccs_valid)) if ccs_valid else -1.0
                theta_deg = float(theta * 180.0 / np.pi)

                dbg = _make_dbg_image_with_footer(
                    crop_bgr, per_point,
                    mean_cc=mean_cc,
                    valid_n=len(P_local),
                    theta_deg=theta_deg,
                    tx=float(tx),
                    tz=float(tz),
                    total_points=total_points,
                )
                cv2.imwrite(
                    os.path.join(self.cfg.debug_out_dir, f"{self.cfg.debug_prefix}_{i:04d}_dbg.png"),
                    dbg
                )
            # -------------------------------------------------------

            # Update reference using stabilized result
            ref_full_gray = to_gray_u8(right_frames[i])
            ref_crop_gray = ref_full_gray[crop_y1:crop_y1 + self.cfg.crop_size,
                                          crop_x1:crop_x1 + self.cfg.crop_size]

        return right_frames, left_frames
