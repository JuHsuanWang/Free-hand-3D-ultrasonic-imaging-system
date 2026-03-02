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

    # --- Precompute gradients for each left frame once (major speedup) ---
    left_grads = []
    for i in range(nL):
        Iy, Ix = np.gradient(left_norm[i])  # note: np.gradient returns (dy, dx)
        left_grads.append((Ix, Iy))

    max_r_ahead = 50
    for l in range(nL):
        ref = left_norm[l]
        r_start = l + 1
        r_end = min(l + 1 + max_r_ahead, nR)  # Python range end is exclusive
        # MATLAB: r starts from l+1 (L0->R1.., L1->R2..)
        for r in range(r_start, r_end):
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

    for i in range(n):
        betas_k = []
        gammas_k = []

        img1 = _to_gray_float64(right_frames[i], cfg)

        for k in range(1, int(cfg.lookahead) + 1):
            j = i + k
            if j >= n:
                break
            img2 = _to_gray_float64(right_frames[j], cfg)

            pos_list = []
            d_list = []

            for (dx, dy) in offsets:
                pcx = cx0 + dx
                pcy = cy0 + dy
                p1 = _extract_patch(img1, pcx, pcy, cell)
                p2 = _extract_patch(img2, pcx, pcy, cell)
                if p1 is None or p2 is None:
                    continue

                vx, vy, _ = _patch_flow_median(p1, p2, cfg)
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
