# algorithms/out_of_plane.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import cv2


@dataclass
class OutOfPlaneConfig:
    # ==== MATLAB params ====
    normalize_brightness: bool = True
    normalize_contrast: bool = True
    target_mean: float = 128.0
    target_std: float = 50.0

    grid_spacing: int = 10
    window_size: int = 25
    max_displacement: float = 150.0

    # ==== MATLAB numeric guards ====
    det_thresh: float = 1e-6


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA/Gray to uint8 gray without per-frame min-max scaling."""
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if g.dtype == np.uint8:
        return g
    # fixed-scale conversion (NOT min-max)
    g = np.clip(g, 0, 255).astype(np.uint8)
    return g


def normalize_image_like_matlab(img_u8: np.ndarray, cfg: OutOfPlaneConfig) -> np.ndarray:
    """
    MATLAB normalizeImage():
      - convert to double
      - if normalize_contrast: z-score, then *target_std + target_mean
      - else if normalize_brightness: shift mean to target_mean
      - clip [0,255]
    """
    img = img_u8.astype(np.float64)

    if cfg.normalize_brightness or cfg.normalize_contrast:
        cur_mean = float(np.mean(img))
        cur_std = float(np.std(img))
        if cfg.normalize_contrast and cur_std > 0:
            normalized = (img - cur_mean) / cur_std
            normalized = normalized * cfg.target_std + cfg.target_mean
        elif cfg.normalize_brightness:
            normalized = img - cur_mean + cfg.target_mean
        else:
            normalized = img
        normalized = np.clip(normalized, 0, 255)
    else:
        normalized = img

    return normalized.astype(np.float64)

def calculate_optical_flow_similarity_like_matlab_boxfilter_pregrad(
    img1: np.ndarray,
    img2: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    points_xy: np.ndarray,
    cfg: OutOfPlaneConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    O(1) LK via boxFilter:
    Replaces per-point window slicing + building A/b with precomputed window sums.

    Keeps the SAME logic as MATLAB-like version:
      - det(AtA) < det_thresh -> skip
      - v = (AtA) \ (Atb)
      - displacement <= max_displacement -> valid
      - residual = ||A v + b|| / len(b), local_sim = exp(-residual)
      - similarity_score = mean(local_sim) over valid points

    Notes:
      - b = -It, It = img2 - img1
      - AtA = [[sum Ix^2, sum IxIy],
               [sum IxIy,  sum Iy^2]]
      - Atb = A^T b = [-sum Ix It, -sum Iy It]
      - SSE = ||A v + b||^2 = v^T AtA v + 2 v^T Atb + (b^T b)
        where b^T b = sum(It^2)
    """
    img1 = img1.astype(np.float64, copy=False)
    img2 = img2.astype(np.float64, copy=False)
    Ix = Ix.astype(np.float64, copy=False)
    Iy = Iy.astype(np.float64, copy=False)

    It = (img2 - img1).astype(np.float64, copy=False)

    h, w = img1.shape
    n = points_xy.shape[0]

    flow_vectors = np.zeros((n, 2), dtype=np.float64)
    valid_mask = np.zeros((n,), dtype=bool)
    disp_mag = np.zeros((n,), dtype=np.float64)

    win = int(cfg.window_size)
    half = win // 2
    area = float(win * win)  # len(b) in MATLAB-like code

    # Use OpenCV boxFilter (C++ optimized) to compute window sums for entire image.
    # ddepth=cv2.CV_64F ensures stable numeric parity with float64 math.
    ksize = (win, win)
    border = cv2.BORDER_CONSTANT

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    Ixt = Ix * It
    Iyt = Iy * It
    It2 = It * It

    Sxx = cv2.boxFilter(Ix2, ddepth=cv2.CV_64F, ksize=ksize, normalize=False, borderType=border)
    Syy = cv2.boxFilter(Iy2, ddepth=cv2.CV_64F, ksize=ksize, normalize=False, borderType=border)
    Sxy = cv2.boxFilter(Ixy, ddepth=cv2.CV_64F, ksize=ksize, normalize=False, borderType=border)
    Sxt = cv2.boxFilter(Ixt, ddepth=cv2.CV_64F, ksize=ksize, normalize=False, borderType=border)
    Syt = cv2.boxFilter(Iyt, ddepth=cv2.CV_64F, ksize=ksize, normalize=False, borderType=border)
    Stt = cv2.boxFilter(It2, ddepth=cv2.CV_64F, ksize=ksize, normalize=False, borderType=border)

    local_sims: List[float] = []

    for i in range(n):
        x = int(points_xy[i, 0])
        y = int(points_xy[i, 1])

        # Match original boundary behavior: require FULL window inside image.
        if (x - half) < 0 or (y - half) < 0 or (x + half) >= w or (y + half) >= h:
            continue

        sxx = float(Sxx[y, x])
        syy = float(Syy[y, x])
        sxy = float(Sxy[y, x])
        sxt = float(Sxt[y, x])
        syt = float(Syt[y, x])
        stt = float(Stt[y, x])

        # det(AtA)
        det = sxx * syy - sxy * sxy
        if det < cfg.det_thresh:
            continue

        # Atb = [-sum(Ix*It), -sum(Iy*It)] = [-sxt, -syt]
        # Solve 2x2 analytically to avoid np.linalg.solve overhead
        bx = -sxt
        by = -syt

        vx = ( syy * bx - sxy * by) / det
        vy = (-sxy * bx + sxx * by) / det

        displacement = float(np.hypot(vx, vy))
        if displacement > cfg.max_displacement:
            continue

        flow_vectors[i, 0] = vx
        flow_vectors[i, 1] = vy
        disp_mag[i] = displacement
        valid_mask[i] = True

        # SSE = v^T AtA v + 2 v^T Atb + b^T b
        # where Atb = [bx, by] and b^T b = sum(It^2) = stt
        vAtAv = (vx * (sxx * vx + sxy * vy) + vy * (sxy * vx + syy * vy))
        vAtb = (vx * bx + vy * by)
        sse = vAtAv + 2.0 * vAtb + stt

        # Guard small negatives due to numeric rounding
        if sse < 0.0:
            sse = 0.0

        residual = float(np.sqrt(sse) / area)
        local_sims.append(float(np.exp(-residual)))

    similarity_score = float(np.mean(local_sims)) if len(local_sims) > 0 else float("nan")
    return flow_vectors, valid_mask, disp_mag, similarity_score

def compute_lr_heatmap_like_matlab(
    cropped_left: np.ndarray,
    cropped_right: np.ndarray,
    cfg: Optional[OutOfPlaneConfig] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """
    MATLAB heatmap loop:
      for l_idx = 1..num_left
        reference_img = all_left_images{l_idx}
        for r_idx = (l_idx + 1)..num_right
          [~, valid_mask, disp_mag, ~] = calculateOpticalFlowSimilarity(...)
          if any(valid) heat(l,r)=mean(disp_mag(valid)) else NaN

    Returns:
      H (nL,nR) with NaN outside computed region
      best list: (l_idx, best_r_idx, min_val) using r>=l+1 only
    """
    if cfg is None:
        cfg = OutOfPlaneConfig()

    if cropped_left is None or cropped_right is None:
        return np.zeros((0, 0), dtype=np.float32), []

    nL = int(len(cropped_left))
    nR = int(len(cropped_right))
    if nL == 0 or nR == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    # build grid points exactly like MATLAB:
    # X_grid = grid_spacing:grid_spacing:(w-grid_spacing)
    # Y_grid = grid_spacing:grid_spacing:(h-grid_spacing)
    ref0 = _to_gray_u8(cropped_left[0])
    h, w = ref0.shape
    xs = np.arange(cfg.grid_spacing, w - cfg.grid_spacing + 1, cfg.grid_spacing)
    ys = np.arange(cfg.grid_spacing, h - cfg.grid_spacing + 1, cfg.grid_spacing)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_points = np.column_stack([Xg.reshape(-1), Yg.reshape(-1)]).astype(np.int32)

    H = np.full((nL, nR), np.nan, dtype=np.float32)

    # pre-normalize all frames (exactly matching MATLAB normalizeImage)
    left_norm = []
    for i in range(nL):
        g = _to_gray_u8(cropped_left[i])
        left_norm.append(normalize_image_like_matlab(g, cfg))

    right_norm = []
    for j in range(nR):
        g = _to_gray_u8(cropped_right[j])
        right_norm.append(normalize_image_like_matlab(g, cfg))

    # --- NEW: precompute gradients for each left frame once (major speedup) ---
    left_grads = []
    for i in range(nL):
        Iy, Ix = np.gradient(left_norm[i])  # note: np.gradient returns (dy, dx)
        left_grads.append((Ix, Iy))


    for l in range(nL):
        ref = left_norm[l]
        # MATLAB: r starts from l+1 (L0->R1.., L1->R2..)
        for r in range(l + 1, nR):
            cur = right_norm[r]
            Ix, Iy = left_grads[l]
            _, valid_mask, disp_mag, _ = calculate_optical_flow_similarity_like_matlab_boxfilter_pregrad(
                ref, cur, Ix, Iy, grid_points, cfg
            )

            if np.any(valid_mask):
                H[l, r] = float(np.mean(disp_mag[valid_mask]))
            else:
                H[l, r] = np.nan

    # red dots: per column (l) take minimal value over r>=l+1
    best: List[Tuple[int, int, float]] = []
    for l in range(nL):
        row = H[l, :].copy()
        row[: min(l + 1, nR)] = np.nan  # enforce r>=l+1
        if not np.any(np.isfinite(row)):
            continue
        r_idx = int(np.nanargmin(row))
        best.append((l, r_idx, float(H[l, r_idx])))

    return H, best
