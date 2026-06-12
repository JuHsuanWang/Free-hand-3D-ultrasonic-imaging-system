# algorithms/out_of_plane.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor
import os

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
    O(1) LK via boxFilter - fully vectorized (no Python loop over grid points).

    Same math / boundary rules as the original MATLAB-like version:
      - det(AtA) < det_thresh -> skip
      - v = inv(AtA) Atb
      - |v| > max_displacement -> skip
      - similarity = mean(exp(-residual)) over valid points
    """
    img1 = img1.astype(np.float64, copy=False)
    img2 = img2.astype(np.float64, copy=False)
    Ix   = Ix.astype(np.float64, copy=False)
    Iy   = Iy.astype(np.float64, copy=False)

    It = (img2 - img1).astype(np.float64, copy=False)
    h, w = img1.shape
    n = points_xy.shape[0]

    flow_vectors = np.zeros((n, 2), dtype=np.float64)
    valid_mask   = np.zeros(n, dtype=bool)
    disp_mag     = np.zeros(n, dtype=np.float64)

    win  = int(cfg.window_size)
    half = win // 2
    area = float(win * win)
    ksize  = (win, win)
    border = cv2.BORDER_CONSTANT

    # --- 6 box-filter maps (3 ref-only + 3 pair-specific) ---
    Sxx = cv2.boxFilter(Ix * Ix, cv2.CV_64F, ksize, normalize=False, borderType=border)
    Syy = cv2.boxFilter(Iy * Iy, cv2.CV_64F, ksize, normalize=False, borderType=border)
    Sxy = cv2.boxFilter(Ix * Iy, cv2.CV_64F, ksize, normalize=False, borderType=border)
    Sxt = cv2.boxFilter(Ix * It, cv2.CV_64F, ksize, normalize=False, borderType=border)
    Syt = cv2.boxFilter(Iy * It, cv2.CV_64F, ksize, normalize=False, borderType=border)
    Stt = cv2.boxFilter(It * It, cv2.CV_64F, ksize, normalize=False, borderType=border)

    # --- vectorized boundary filter ---
    xs = points_xy[:, 0].astype(np.int32)
    ys = points_xy[:, 1].astype(np.int32)
    inb = (xs - half >= 0) & (ys - half >= 0) & (xs + half < w) & (ys + half < h)
    if not np.any(inb):
        return flow_vectors, valid_mask, disp_mag, float("nan")

    xi = xs[inb];  yi = ys[inb]

    sxx = Sxx[yi, xi];  syy = Syy[yi, xi];  sxy = Sxy[yi, xi]
    sxt = Sxt[yi, xi];  syt = Syt[yi, xi];  stt = Stt[yi, xi]

    det  = sxx * syy - sxy * sxy
    good = det >= float(cfg.det_thresh)

    bx = -sxt;  by = -syt
    ds = np.where(good, det, 1.0)          # safe divisor
    vx = np.where(good, (syy * bx - sxy * by) / ds,  0.0)
    vy = np.where(good, (-sxy * bx + sxx * by) / ds, 0.0)

    disp = np.hypot(vx, vy)
    good2 = good & (disp <= float(cfg.max_displacement))

    # SSE -> local similarity
    vAtAv = vx * (sxx * vx + sxy * vy) + vy * (sxy * vx + syy * vy)
    vAtb  = vx * bx + vy * by
    sse   = np.maximum(vAtAv + 2.0 * vAtb + stt, 0.0)
    local_sim = np.exp(-np.sqrt(sse) / area)

    # scatter results back into full-length arrays
    inb_idx = np.where(inb)[0]
    valid_idx = inb_idx[good2]
    flow_vectors[valid_idx, 0] = vx[good2]
    flow_vectors[valid_idx, 1] = vy[good2]
    disp_mag[valid_idx]        = disp[good2]
    valid_mask[valid_idx]      = True

    similarity_score = float(np.mean(local_sim[good2])) if np.any(good2) else float("nan")
    return flow_vectors, valid_mask, disp_mag, similarity_score


# ---------------------------------------------------------------------------
# Fast-path helpers for heatmap computation
# ---------------------------------------------------------------------------

def _precompute_ref_maps(
    Ix: np.ndarray, Iy: np.ndarray, win: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Sxx, Syy, Sxy (the AtA box-filter maps) for ONE reference frame.
    These only depend on the reference image, so they can be cached and reused
    for every comparison frame - saving 3 boxFilter calls per pair.
    """
    ksize = (win, win)
    b = cv2.BORDER_CONSTANT
    Sxx = cv2.boxFilter(Ix * Ix, cv2.CV_64F, ksize, normalize=False, borderType=b)
    Syy = cv2.boxFilter(Iy * Iy, cv2.CV_64F, ksize, normalize=False, borderType=b)
    Sxy = cv2.boxFilter(Ix * Iy, cv2.CV_64F, ksize, normalize=False, borderType=b)
    return Sxx, Syy, Sxy


def _lk_mean_disp_fast(
    ref_norm: np.ndarray,
    cur_norm: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Sxy: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    win: int,
    det_thresh: float,
    max_displacement: float,
) -> float:
    """
    Vectorized mean LK displacement at pre-filtered grid points.

    Sxx/Syy/Sxy are precomputed for the reference frame (cached by caller).
    Only the pair-specific maps Sxt, Syt are computed here (2 boxFilters instead of 6).
    Grid point sampling is fully vectorized - no Python loop.
    """
    It = cur_norm - ref_norm
    ksize = (win, win)
    b = cv2.BORDER_CONSTANT
    Sxt = cv2.boxFilter(Ix * It, cv2.CV_64F, ksize, normalize=False, borderType=b)
    Syt = cv2.boxFilter(Iy * It, cv2.CV_64F, ksize, normalize=False, borderType=b)

    sxx = Sxx[pts_y, pts_x];  syy = Syy[pts_y, pts_x];  sxy = Sxy[pts_y, pts_x]
    bx  = -Sxt[pts_y, pts_x]; by  = -Syt[pts_y, pts_x]

    det  = sxx * syy - sxy * sxy
    good = det >= det_thresh
    if not np.any(good):
        return float("nan")

    ds = np.where(good, det, 1.0)
    vx = np.where(good, (syy * bx - sxy * by) / ds,  0.0)
    vy = np.where(good, (-sxy * bx + sxx * by) / ds, 0.0)

    disp  = np.hypot(vx, vy)
    valid = good & (disp <= max_displacement)
    return float(np.mean(disp[valid])) if np.any(valid) else float("nan")

def _normalize_frame_sequence(
    frames: np.ndarray,
    cfg: OutOfPlaneConfig,
) -> List[np.ndarray]:
    return [normalize_image_like_matlab(_to_gray_u8(frames[i]), cfg) for i in range(int(len(frames)))]


def _compute_lr_heatmap_from_precomputed(
    ref_norm: Sequence[np.ndarray],
    cmp_norm: Sequence[np.ndarray],
    cfg: OutOfPlaneConfig,
    max_r_ahead: int,
    frame_stride: int,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    n_ref = int(len(ref_norm))
    n_cmp = int(len(cmp_norm))
    if n_ref == 0 or n_cmp == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    h, w = ref_norm[0].shape
    xs = np.arange(cfg.grid_spacing, w - cfg.grid_spacing + 1, cfg.grid_spacing)
    ys = np.arange(cfg.grid_spacing, h - cfg.grid_spacing + 1, cfg.grid_spacing)
    Xg, Yg = np.meshgrid(xs, ys)
    all_pts = np.column_stack([Xg.reshape(-1), Yg.reshape(-1)]).astype(np.int32)

    half = int(cfg.window_size) // 2
    inb = ((all_pts[:, 0] - half >= 0) & (all_pts[:, 1] - half >= 0) &
           (all_pts[:, 0] + half < w)  & (all_pts[:, 1] + half < h))
    pts_x = all_pts[inb, 0].astype(np.int32)
    pts_y = all_pts[inb, 1].astype(np.int32)

    H = np.full((n_ref, n_cmp), np.nan, dtype=np.float32)
    if len(pts_x) == 0:
        return H, []

    ref_grads = []
    for i in range(n_ref):
        Iy, Ix = np.gradient(ref_norm[i])
        ref_grads.append((Ix, Iy))

    win = int(cfg.window_size)
    det_thresh = float(cfg.det_thresh)
    max_disp = float(cfg.max_displacement)
    stride = max(1, int(frame_stride))
    rows = list(range(0, n_ref, stride))

    def _compute_row(l: int) -> Tuple[int, np.ndarray]:
        row = np.full((n_cmp,), np.nan, dtype=np.float32)
        Ix, Iy = ref_grads[l]
        Sxx, Syy, Sxy = _precompute_ref_maps(Ix, Iy, win)
        ref = ref_norm[l]
        r_end = min(l + 1 + max_r_ahead, n_cmp)
        for r in range(l + 1, r_end):
            row[r] = _lk_mean_disp_fast(
                ref, cmp_norm[r], Ix, Iy,
                Sxx, Syy, Sxy,
                pts_x, pts_y,
                win, det_thresh, max_disp,
            )
        return l, row

    if len(rows) > 1:
        max_workers = min(len(rows), max(1, int(os.cpu_count() or 1)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for l, row in ex.map(_compute_row, rows):
                H[l, :] = row
    else:
        for l in rows:
            _, row = _compute_row(l)
            H[l, :] = row

    best: List[Tuple[int, int, float]] = []
    for l in rows:
        row = H[l, :].copy()
        row[: min(l + 1, n_cmp)] = np.nan
        if not np.any(np.isfinite(row)):
            continue
        r_idx = int(np.nanargmin(row))
        best.append((l, r_idx, float(H[l, r_idx])))

    return H, best


def compute_lr_heatmap_like_matlab(
    cropped_left: np.ndarray,
    cropped_right: np.ndarray,
    cfg: Optional[OutOfPlaneConfig] = None,
    max_r_ahead: int = 50,
    frame_stride: int = 1,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """
    Optimised MATLAB-style heatmap:
      H[l, r] = mean LK displacement(left_l -> right_r)  for r in [l+1, l+max_r_ahead)

    Speed improvements over the original:
      - Sxx/Syy/Sxy (AtA maps) cached once per reference frame -> 3 fewer boxFilters/pair
      - Grid-point sampling fully vectorised -> no Python loop per pair
      - Boundary check done once at startup
      - Optional frame_stride to compute every Nth reference frame (coarser but faster)

    Returns:
      H (nL, nR) float32, NaN for uncomputed cells
      best: [(l, best_r, min_val), ...] - one entry per reference frame
    """
    if cfg is None:
        cfg = OutOfPlaneConfig()

    if cropped_left is None or cropped_right is None:
        return np.zeros((0, 0), dtype=np.float32), []

    nL = int(len(cropped_left))
    nR = int(len(cropped_right))
    if nL == 0 or nR == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    left_norm = _normalize_frame_sequence(cropped_left, cfg)
    right_norm = _normalize_frame_sequence(cropped_right, cfg)
    return _compute_lr_heatmap_from_precomputed(
        left_norm, right_norm, cfg,
        max_r_ahead=max_r_ahead,
        frame_stride=frame_stride,
    )


def _compute_r2_from_best_pairs(best: List[Tuple[int, int, float]]) -> float:
    """Compute R^2 of a robust linear regression through best-pair (ref, cmp) coordinates."""
    if len(best) < 3:
        return float("nan")
    xs = np.array([p[0] for p in best], dtype=np.float64)
    ys = np.array([p[1] for p in best], dtype=np.float64)
    if len(xs) >= 5:
        a0, b0 = np.polyfit(xs, ys, 1)
        res = ys - (a0 * xs + b0)
        med = np.median(res)
        mad = float(np.median(np.abs(res - med))) + 1e-6
        inl = np.abs(res - med) < 2.0 * mad
        if np.sum(inl) >= 3:
            xs, ys = xs[inl], ys[inl]
    a, b = np.polyfit(xs, ys, 1)
    y_fit = a * xs + b
    ss_res = float(np.sum((ys - y_fit) ** 2))
    ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_scan_direction_heatmaps(
    cropped_left: np.ndarray,
    cropped_right: np.ndarray,
    cfg: Optional[OutOfPlaneConfig] = None,
    max_r_ahead: int = 50,
    frame_stride: int = 1,
) -> Tuple[str, np.ndarray, List, np.ndarray, List, float, float]:
    """
    Compute both forward (L->R) and reverse (R->L) heatmaps and auto-detect scan direction.

    Forward: left frames as reference, right frames as comparison (r >= l+1).
    Reverse: right frames as reference, left frames as comparison (r >= l+1).

    The direction whose best-pair regression yields a higher R^2 is selected.

    Returns:
        direction   : "forward" or "reverse"
        H_fwd       : (nL, nR) forward heatmap
        best_fwd    : forward best pairs [(l, r, val), ...]
        H_rev       : (nR, nL) reverse heatmap
        best_rev    : reverse best pairs [(r, l, val), ...]
        r2_fwd      : R^2 of forward regression
        r2_rev      : R^2 of reverse regression
    """
    if cfg is None:
        cfg = OutOfPlaneConfig()

    left_norm = _normalize_frame_sequence(cropped_left, cfg)
    right_norm = _normalize_frame_sequence(cropped_right, cfg)

    H_fwd, best_fwd = _compute_lr_heatmap_from_precomputed(
        left_norm, right_norm, cfg,
        max_r_ahead=max_r_ahead,
        frame_stride=frame_stride,
    )
    H_rev, best_rev = _compute_lr_heatmap_from_precomputed(
        right_norm, left_norm, cfg,
        max_r_ahead=max_r_ahead,
        frame_stride=frame_stride,
    )

    r2_fwd = _compute_r2_from_best_pairs(best_fwd)
    r2_rev = _compute_r2_from_best_pairs(best_rev)

    print(f"[ScanDir] R2_fwd={r2_fwd:.3f}, R2_rev={r2_rev:.3f}")

    if np.isnan(r2_fwd) and np.isnan(r2_rev):
        direction = "forward"
    elif np.isnan(r2_rev) or (not np.isnan(r2_fwd) and r2_fwd >= r2_rev):
        direction = "forward"
    else:
        direction = "reverse"

    print(f"[ScanDir] Detected: {direction}")
    return direction, H_fwd, best_fwd, H_rev, best_rev, r2_fwd, r2_rev

# ============================================================
# Out-of-plane rotation (beta/gamma) from 3x3 grid (exclude center)
# ============================================================

@dataclass
class OutOfPlaneRotConfig:
    # LK params (reuse the same LK core)
    normalize_brightness: bool = False
    normalize_contrast: bool = False
    target_mean: float = 128.0
    target_std: float = 50.0

    grid_spacing: int = 10
    window_size: int = 25
    max_displacement: float = 150.0
    det_thresh: float = 1e-6

    # motion aggregation
    lookahead: int = 5                 # compare frame i with i+1..i+lookahead
    enable_time_median_filter: bool = True
    time_median_win: int = 5           # temporal median filter window (odd suggested)

    # 3x3 cell geometry (in ROI pixel coordinates)
    cell_size: int = 100               # each cell is 100x100 in your UI
    exclude_center: bool = True        # use 1,2,3,4,6,7,8,9

    # minimum patches required to fit deformation
    min_patches_for_fit: int = 4


def _median_filter_1d_nan(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or win <= 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    r = win // 2
    y = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(n):
        a = max(0, i - r)
        b = min(n, i + r + 1)
        w = x[a:b]
        w = w[np.isfinite(w)]
        if w.size > 0:
            y[i] = float(np.median(w))
    return y


def _interp_extrap_1d_nan(x: np.ndarray) -> np.ndarray:
    """
    Linear interpolation for NaNs inside range + linear extrapolation at both ends.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x
    idx = np.arange(n, dtype=np.float64)
    m = np.isfinite(x)
    if np.all(m):
        return x.copy()
    if np.sum(m) == 0:
        return np.zeros_like(x, dtype=np.float64)

    y = x.copy()

    # interp inside
    y[~m] = np.interp(idx[~m], idx[m], x[m])

    # extrap left using first two finite points
    finite_idx = idx[m]
    finite_val = x[m]
    if finite_idx.size >= 2:
        i0, i1 = finite_idx[0], finite_idx[1]
        v0, v1 = finite_val[0], finite_val[1]
        slope = (v1 - v0) / (i1 - i0 + 1e-12)
        left = np.where(idx < i0)[0]
        y[left] = v0 + slope * (idx[left] - i0)

        # extrap right using last two finite points
        j0, j1 = finite_idx[-2], finite_idx[-1]
        u0, u1 = finite_val[-2], finite_val[-1]
        slope_r = (u1 - u0) / (j1 - j0 + 1e-12)
        right = np.where(idx > j1)[0]
        y[right] = u1 + slope_r * (idx[right] - j1)

    else:
        # only one finite point -> fill all with it
        y[:] = finite_val[0]

    return y


def _to_gray_float64(img: np.ndarray, cfg: OutOfPlaneRotConfig) -> np.ndarray:
    g_u8 = _to_gray_u8(img)
    if cfg.normalize_brightness or cfg.normalize_contrast:
        return normalize_image_like_matlab(g_u8, OutOfPlaneConfig(
            normalize_brightness=cfg.normalize_brightness,
            normalize_contrast=cfg.normalize_contrast,
            target_mean=cfg.target_mean,
            target_std=cfg.target_std,
            grid_spacing=cfg.grid_spacing,
            window_size=cfg.window_size,
            max_displacement=cfg.max_displacement,
            det_thresh=cfg.det_thresh,
        ))
    return g_u8.astype(np.float64)


def _extract_patch(gray: np.ndarray, cx: int, cy: int, cell: int) -> Optional[np.ndarray]:
    half = cell // 2
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    h, w = gray.shape[:2]
    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
        return None
    return gray[y1:y2, x1:x2]


def _patch_flow_median(
    img1_gray: np.ndarray,
    img2_gray: np.ndarray,
    cfg: OutOfPlaneRotConfig
) -> Tuple[float, float, float]:
    """
    Return (vx_med, vy_med, valid_ratio).
    """
    h, w = img1_gray.shape
    xs = np.arange(cfg.grid_spacing, w - cfg.grid_spacing + 1, cfg.grid_spacing)
    ys = np.arange(cfg.grid_spacing, h - cfg.grid_spacing + 1, cfg.grid_spacing)
    if xs.size == 0 or ys.size == 0:
        return np.nan, np.nan, 0.0

    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.column_stack([Xg.reshape(-1), Yg.reshape(-1)]).astype(np.int32)

    Iy, Ix = np.gradient(img1_gray)  # (dy, dx)
    of_cfg = OutOfPlaneConfig(
        normalize_brightness=False,
        normalize_contrast=False,
        target_mean=cfg.target_mean,
        target_std=cfg.target_std,
        grid_spacing=cfg.grid_spacing,
        window_size=cfg.window_size,
        max_displacement=cfg.max_displacement,
        det_thresh=cfg.det_thresh,
    )

    flow, valid_mask, _, _ = calculate_optical_flow_similarity_like_matlab_boxfilter_pregrad(
        img1_gray, img2_gray, Ix, Iy, pts, of_cfg
    )
    if not np.any(valid_mask):
        return np.nan, np.nan, 0.0

    v = flow[valid_mask]
    vx = float(np.median(v[:, 0]))
    vy = float(np.median(v[:, 1]))
    return vx, vy, float(np.mean(valid_mask))


def _patch_ref_cache(
    img1_gray: np.ndarray,
    cfg: OutOfPlaneRotConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    h, w = img1_gray.shape
    xs = np.arange(cfg.grid_spacing, w - cfg.grid_spacing + 1, cfg.grid_spacing)
    ys = np.arange(cfg.grid_spacing, h - cfg.grid_spacing + 1, cfg.grid_spacing)
    if xs.size == 0 or ys.size == 0:
        return None

    Xg, Yg = np.meshgrid(xs, ys)
    all_pts = np.column_stack([Xg.reshape(-1), Yg.reshape(-1)]).astype(np.int32)

    half = int(cfg.window_size) // 2
    inb = ((all_pts[:, 0] - half >= 0) & (all_pts[:, 1] - half >= 0) &
           (all_pts[:, 0] + half < w) & (all_pts[:, 1] + half < h))
    pts_x = all_pts[inb, 0].astype(np.int32)
    pts_y = all_pts[inb, 1].astype(np.int32)
    if pts_x.size == 0:
        return None

    Iy, Ix = np.gradient(img1_gray)
    Sxx, Syy, Sxy = _precompute_ref_maps(Ix, Iy, int(cfg.window_size))
    return Ix, Iy, Sxx, Syy, Sxy, pts_x, pts_y


def _patch_flow_median_from_cache(
    img1_gray: np.ndarray,
    img2_gray: np.ndarray,
    cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cfg: OutOfPlaneRotConfig,
) -> Tuple[float, float, float]:
    Ix, Iy, Sxx, Syy, Sxy, pts_x, pts_y = cache
    It = img2_gray - img1_gray
    ksize = (int(cfg.window_size), int(cfg.window_size))
    b = cv2.BORDER_CONSTANT
    Sxt = cv2.boxFilter(Ix * It, cv2.CV_64F, ksize, normalize=False, borderType=b)
    Syt = cv2.boxFilter(Iy * It, cv2.CV_64F, ksize, normalize=False, borderType=b)

    sxx = Sxx[pts_y, pts_x]; syy = Syy[pts_y, pts_x]; sxy = Sxy[pts_y, pts_x]
    bx = -Sxt[pts_y, pts_x]; by = -Syt[pts_y, pts_x]

    det = sxx * syy - sxy * sxy
    good = det >= float(cfg.det_thresh)
    if not np.any(good):
        return np.nan, np.nan, 0.0

    ds = np.where(good, det, 1.0)
    vx = np.where(good, (syy * bx - sxy * by) / ds, 0.0)
    vy = np.where(good, (-sxy * bx + sxx * by) / ds, 0.0)

    disp = np.hypot(vx, vy)
    valid = good & (disp <= float(cfg.max_displacement))
    if not np.any(valid):
        return np.nan, np.nan, 0.0

    return float(np.median(vx[valid])), float(np.median(vy[valid])), float(np.mean(valid))


def _fit_affine_from_patch_flows(pos: np.ndarray, d: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit d = A @ pos + t , where pos=(N,2), d=(N,2).
    Return A (2,2). t is ignored for beta/gamma extraction.
    """
    if pos.shape[0] < 3:
        return None
    # design matrix for [x z 1]
    X = np.column_stack([pos[:, 0], pos[:, 1], np.ones((pos.shape[0],), dtype=np.float64)])  # (N,3)
    # solve for dx, dz separately
    try:
        px, *_ = np.linalg.lstsq(X, d[:, 0], rcond=None)  # (3,)
        py, *_ = np.linalg.lstsq(X, d[:, 1], rcond=None)  # (3,)
    except Exception:
        return None

    # A = [[px0, px1],
    #      [py0, py1]]
    A = np.array([[px[0], px[1]],
                  [py[0], py[1]]], dtype=np.float64)
    return A


def compute_beta_gamma_from_right_grid(
    right_frames: np.ndarray,
    click_point_xy: Tuple[int, int],
    cfg: Optional[OutOfPlaneRotConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use R-plane 3x3 grid (exclude center) patches:
      cells 1,2,3,4,6,7,8,9
    For each frame i:
      compare i vs i+1..i+lookahead
      each comparison:
        - compute LK flow in each patch (median of valid flows)
        - fit an affine deformation model: d = A*[x,z] + t (pos relative to center)
        - extract beta/gamma as shear proxies:
            beta  = atan(A[1,0])  (vertical displacement depends on x)
            gamma = atan(A[0,1])  (horizontal displacement depends on z)
      aggregate across lookahead with median
    Then temporal median filter + interp/extrap to fill NaNs.

    NOTE:
    - This gives a stable "rotation proxy" series aligned with your MATLAB-style pipeline.
    - Absolute physical calibration (deg/mm) is not applied here; angles are derived from deformation slopes.
    """
    if cfg is None:
        cfg = OutOfPlaneRotConfig()

    if right_frames is None or len(right_frames) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    n = int(len(right_frames))
    cx0, cy0 = int(click_point_xy[0]), int(click_point_xy[1])
    cell = int(cfg.cell_size)

    # 3x3 cell center offsets (dx,dy): numbers follow
    # 1 2 3
    # 4 5 6
    # 7 8 9
    offsets = []
    for ry, dy in enumerate([-cell, 0, cell]):
        for rx, dx in enumerate([-cell, 0, cell]):
            if cfg.exclude_center and dx == 0 and dy == 0:
                continue
            offsets.append((dx, dy))
    offsets = offsets  # length 8 when exclude_center=True

    beta = np.full((n,), np.nan, dtype=np.float64)
    gamma = np.full((n,), np.nan, dtype=np.float64)
    gray_frames = [_to_gray_float64(right_frames[i], cfg) for i in range(n)]

    for i in range(n):
        betas_k = []
        gammas_k = []

        img1 = gray_frames[i]
        patch_cache = {}
        for offset in offsets:
            dx, dy = offset
            p1 = _extract_patch(img1, cx0 + dx, cy0 + dy, cell)
            if p1 is None:
                continue
            cache = _patch_ref_cache(p1, cfg)
            if cache is not None:
                patch_cache[offset] = (p1, cache)

        for k in range(1, int(cfg.lookahead) + 1):
            j = i + k
            if j >= n:
                break
            img2 = gray_frames[j]

            pos_list = []
            d_list = []

            for (dx, dy) in offsets:
                cached = patch_cache.get((dx, dy))
                if cached is None:
                    continue
                p1, cache = cached
                p2 = _extract_patch(img2, cx0 + dx, cy0 + dy, cell)
                if p2 is None:
                    continue

                vx, vy, _ = _patch_flow_median_from_cache(p1, p2, cache, cfg)
                if not np.isfinite(vx) or not np.isfinite(vy):
                    continue

                # position relative to center (use (x,z) convention: x=horizontal, z=vertical)
                pos_list.append([float(dx), float(dy)])
                d_list.append([float(vx), float(vy)])

            if len(pos_list) < int(cfg.min_patches_for_fit):
                continue

            pos_arr = np.asarray(pos_list, dtype=np.float64)
            d_arr = np.asarray(d_list, dtype=np.float64)

            A = _fit_affine_from_patch_flows(pos_arr, d_arr)
            if A is None:
                continue

            # shear-based angle proxies (small-angle)
            beta_k = float(np.degrees(np.arctan(A[1, 0])))
            gamma_k = float(np.degrees(np.arctan(A[0, 1])))

            betas_k.append(beta_k)
            gammas_k.append(gamma_k)

        if len(betas_k) > 0:
            beta[i] = float(np.median(betas_k))
        if len(gammas_k) > 0:
            gamma[i] = float(np.median(gammas_k))

        # temporal median filter (optional)
        if bool(getattr(cfg, "enable_time_median_filter", True)) and int(cfg.time_median_win) > 1:
            beta_f = _median_filter_1d_nan(beta, int(cfg.time_median_win))
            gamma_f = _median_filter_1d_nan(gamma, int(cfg.time_median_win))
        else:
            beta_f = beta.copy()
            gamma_f = gamma.copy()


    # interp/extrap NaNs
    beta_out = _interp_extrap_1d_nan(beta_f)
    gamma_out = _interp_extrap_1d_nan(gamma_f)

    return beta_out, gamma_out
