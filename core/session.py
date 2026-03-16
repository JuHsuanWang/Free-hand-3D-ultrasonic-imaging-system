# core/session.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np


@dataclass
class SessionState:
    """Stores app state and shared data between backend and GUI."""

    # Input mode / paths
    input_mode: str = "video"   # "video" | "simulation" | "live"
    left_video_path: str = ""
    right_video_path: str = ""
    left_source_path: str = ""
    right_source_path: str = ""

    # Original frames (BGR)
    left_frames_original: Optional[np.ndarray] = None
    right_frames_original: Optional[np.ndarray] = None

    # ROI-cropped frames (BGR or BGRA after stabilization)
    left_frames: Optional[np.ndarray] = None
    right_frames: Optional[np.ndarray] = None

    # 100x100 crops around click point (BGR/BGRA)
    cropped_left: Optional[np.ndarray] = None
    cropped_right: Optional[np.ndarray] = None

    # For LK Y-heatmap: full-width band crops
    band_left: Optional[np.ndarray] = None   # (N, band_h, frame_w, C)
    band_right: Optional[np.ndarray] = None  # (N, band_h, frame_w, C)

    # ROI selection
    roi_pt1: Optional[Tuple[int, int]] = None  # (x,y)
    roi_pt2: Optional[Tuple[int, int]] = None  # (x,y)
    roi_selection_phase: int = 1  # 1: pt1, 2: pt2
    roi_confirmed: bool = False

    # Crop center selection in ROI coordinates
    click_point: Optional[Tuple[int, int]] = None
    selection_confirmed: bool = False

    # -------------------------
    # ROI depth calibration + LR distance mapping
    # -------------------------
    # Assumption: within the ROI (cyan box), the top row corresponds to 0 mm depth
    # and the bottom row corresponds to roi_depth_mm.
    roi_depth_mm: float = 50.0
    depth_mm_per_px: float = 0.0
    click_depth_mm: Optional[float] = None

    # Frame dimensions
    orig_frame_h: int = 0
    orig_frame_w: int = 0
    frame_h: int = 0
    frame_w: int = 0
        # Depth->LR lookup table (from your calibration)
    # Piecewise-linear interpolation will be used (NOT global regression).
    lr_lut_depth_mm: Tuple[float, ...] = (
        10, 20, 25, 30, 33, 35, 37, 40, 42, 45, 47, 50, 52, 55, 60, 62
    )
    lr_lut_value: Tuple[float, ...] = (
        3.068348694, 1.512566257, 1.253269185, 0.993972112,
        0.648242682, 0.432161788, 0.475377967, 0.259297073,
        0.129648536, 0.086432358, 0.086432358, -0.129648536,
        -0.388945609, -0.432161788, -0.691458861, -0.691458861
    )
    click_lr_distance: Optional[float] = None


    # Segmentation results: list of (M,2) points for each frame
    contour_points_list: List[np.ndarray] = field(default_factory=list)

    # --- NEW: Manual Labeling Storage ---
    # Stores the user-drawn contours for specific frames.
    # Key: frame_index (int)
    # Value: List of (u, v) tuples representing 2D image coordinates.
    # These coordinates are independent of 3D rotation (beta/gamma).
    manual_contours: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)

    # -------------------------
    # Out-of-plane (Y) heatmap results
    # -------------------------
    y_heatmap: Optional[np.ndarray] = None              # (nL, nR), float32, NaN for invalid
    y_best_pairs: List[Tuple[int, int, float]] = field(default_factory=list)  # (l_idx, r_idx, min_val)


    # Visualization toggles
    frames_visible: bool = True
    point_cloud_visible: bool = True

    # 3D overlay toggles
    crop_box_visible: bool = True     # red box
    band_box_visible: bool = True     # yellow band
    grid9_visible: bool = True       # orange 3x3 grid


    # Stabilization transforms (optional debug record)
    stabilize_transforms: list = field(default_factory=list)

    # --- NEW: per-frame relative motion from freehand (i -> i+1) ---
    fh_dx_mm: Optional[np.ndarray] = None      # shape (n-1,)
    fh_dz_mm: Optional[np.ndarray] = None      # shape (n-1,)
    fh_dalpha_deg: Optional[np.ndarray] = None # shape (n-1,)
    fh_dy_mm: Optional[np.ndarray] = None      # shape (n-1,) constant for now
    # --------------------------------------------------------------
    # Out-of-plane rotation (beta/gamma) results
    # -------------------------
    beta_deg: Optional[np.ndarray] = None   # per-frame beta (deg)
    gamma_deg: Optional[np.ndarray] = None  # per-frame gamma (deg)

    # --- Volume measurement ---
    surface_mesh_px: Optional[object] = None   # display mesh in current pixel-based world
    surface_mesh_mm: Optional[object] = None   # copied/scaled mesh for volume calculation
    surface_volume_mm3: Optional[float] = None
    surface_volume_ml: Optional[float] = None
    # -------------------------



    def ensure_original_dims(self):
        """Populate original frame size from right_frames_original."""
        if self.right_frames_original is None or len(self.right_frames_original) == 0:
            raise ValueError("No frames loaded for right video.")
        self.orig_frame_h, self.orig_frame_w = self.right_frames_original[0].shape[:2]

    def ensure_roi_dims(self):
        """Populate ROI frame size from right_frames."""
        if self.right_frames is None or len(self.right_frames) == 0:
            raise ValueError("No ROI-cropped frames available.")
        self.frame_h, self.frame_w = self.right_frames[0].shape[:2]
        # Update mm-per-pixel calibration for depth (vertical axis) in ROI.
        # Use (frame_h - 1) so that bottom-most pixel maps close to roi_depth_mm.
        denom = max(1, int(self.frame_h) - 1)
        self.depth_mm_per_px = float(self.roi_depth_mm) / float(denom)

    def depth_to_lr(self, depth_mm: float) -> float:
        """
        Piecewise-linear interpolation with linear extrapolation
        at both ends of the LUT.
        """
        d = float(depth_mm)
        xs = self.lr_lut_depth_mm
        ys = self.lr_lut_value

        # ---- Extrapolation below minimum depth ----
        if d <= xs[0]:
            x0, x1 = float(xs[0]), float(xs[1])
            y0, y1 = float(ys[0]), float(ys[1])
            slope = (y1 - y0) / (x1 - x0)
            return y0 + slope * (d - x0)

        # ---- Extrapolation above maximum depth ----
        if d >= xs[-1]:
            x0, x1 = float(xs[-2]), float(xs[-1])
            y0, y1 = float(ys[-2]), float(ys[-1])
            slope = (y1 - y0) / (x1 - x0)
            return y1 + slope * (d - x1)

        # ---- Interpolation inside LUT ----
        for i in range(len(xs) - 1):
            x0, x1 = float(xs[i]), float(xs[i + 1])
            if x0 <= d <= x1:
                y0, y1 = float(ys[i]), float(ys[i + 1])
                t = (d - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        # Safety fallback
        return float(ys[-1])

    def update_click_metrics(self, img_y: int) -> None:
        """Update click_depth_mm and click_lr_distance from a ROI y-coordinate (pixel)."""
        self.click_depth_mm = float(img_y) * float(self.depth_mm_per_px)
        self.click_lr_distance = self.depth_to_lr(self.click_depth_mm)



