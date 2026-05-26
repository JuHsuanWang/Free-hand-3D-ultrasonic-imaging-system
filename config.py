# config.py
from dataclasses import dataclass
import multiprocessing


@dataclass
class AppConfig:
    """Centralized configuration for the whole app."""

    # --------------------------------------------------
    # Runtime mode
    # --------------------------------------------------
    input_mode: str = "video"   # "video" | "simulation" | "live"

    # Output root
    output_root: str = "output"
    run_name: str = "run"  # will be overwritten in main.py based on filename

    # --------------------------------------------------
    # Video sampling
    # --------------------------------------------------
    output_fps: int = 10

    # --------------------------------------------------
    # Default runtime values (these will be overwritten by apply_mode_settings)
    # --------------------------------------------------
    crop_size: int = 100
    y_spacing: float = 5.4
    fh_dy_mm_per_frame: float = 0.1950

    # --------------------------------------------------
    # Per-mode settings
    # --------------------------------------------------
    # Offline video mode
    video_crop_size: int = 100
    #video_y_spacing: float = 5.4
    video_y_spacing: float = 3.4
    #video_fh_dy_mm_per_frame: float = 0.1950
    video_fh_dy_mm_per_frame: float = 0.11
    video_enable_stabilization: bool = True
    video_stab_win: int = 16
    video_stab_search: int = 4
    video_stab_cc_thresh: float = 0.2

    # Simulation mode
    sim_crop_size: int = 60
    sim_y_spacing: float = 1.0
    sim_fh_dy_mm_per_frame: float = 0.1
    sim_enable_stabilization: bool = False
    sim_stab_win: int = 16
    sim_stab_search: int = 4
    sim_stab_cc_thresh: float = 0.2
    sim_auto_threshold: int = 100  # Intensity < this is target (for simulation auto-labeling)
    sim_min_area: int = 500

    # Live capture mode
    live_crop_size: int = 40
    live_y_spacing: float = 1.1
    live_fh_dy_mm_per_frame: float = 0.1
    live_enable_stabilization: bool = True
    live_stab_win: int = 16
    live_stab_search: int = 4
    live_stab_cc_thresh: float = 0.2

    # --------------------------------------------------
    # Segmentation settings
    # --------------------------------------------------
    bright_low: int = 140     # keep pixels >= bright_low
    bright_high: int = 220    # keep pixels <= bright_high
    min_contour_area: int = 2000

    # Performance
    num_workers: int = multiprocessing.cpu_count()

    # Output (will be overwritten to output/<run_name>/...)
    save_png: bool = False
    png_out_dir: str = "cropped_frames"

    # Stabilization
    enable_stabilization: bool = True
    stabilized_out_dir: str = "stabilized_frames"

    # Stabilization tracking parameters (NCC + Kabsch)
    stab_grid: int = 5
    stab_win: int = 16
    stab_search: int = 4
    stab_cc_thresh: float = 0.2

    # Pixel spacing in mm (anisotropic allowed)
    dx_mm: float = 0.1
    dz_mm: float = 0.1

    # Stabilization debug
    save_stab_debug: bool = False

    # When True, detect segmented forward/reverse scan direction.
    # When False, use the original fixed L(reference) -> R(comparison) series.
    enable_scan_direction_detection: bool = False

    # Scan direction is allowed to change only at coarse segment boundaries.
    scan_direction_segment_frames: int = 100
    y_heatmap_max_r_ahead: int = 50
    y_heatmap_ignore_tail_frames: int = 50

    # --- Out-of-plane rotation (beta/gamma) post-processing ---
    enable_beta_gamma_median_filter: bool = False
    beta_gamma_median_win: int = 5

    crop_top_frac: float = 0.25
    enable_crop_top: bool = True

    # --- Surface post-processing ---
    enable_surface_smoothing: bool = True
    surface_smoothing_iterations: int = 30
    surface_smoothing_relaxation: float = 0.01

    # --- Freehand pose export in EM tracker-like CSV format ---
    enable_tracker_like_export: bool = False
    tracker_like_export_filename: str = "freehand_tracker_like.csv"
    tracker_like_port: str = "Port:11"

    # --- Evaluation export (Python -> MATLAB) ---
    enable_eval_export: bool = True
    eval_export_stride: int = 10
    eval_export_filename: str = "pred_eval.mat"
    # "display" exports the same shape shown by Generate 3D:
    #   X/Z in ROI pixels, Y in video_y_spacing display units.
    # "display_scaled_mm" preserves that display shape, then uniformly scales
    #   X/Y/Z by calibrated mm-per-pixel so it can be compared to mm GT.
    # "physical_mm" exports physically scaled coordinates:
    #   X/Z in calibrated mm, Y in video_fh_dy_mm_per_frame mm.
    eval_export_space: str = "display_scaled_mm"
    # Y origin used only in pred_eval.mat export:
    # "labeled_mid" -> midpoint of labeled frames becomes y=0
    # "scan_mid"    -> midpoint of all loaded frames becomes y=0
    # "frame0"      -> frame 0 stays y=0
    eval_y_origin_mode: str = "labeled_mid"
    # Set >= 0 to force a specific 0-based frame index as y=0.
    eval_y_origin_frame_idx0: int = -1

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def apply_mode_settings(self, mode: str) -> None:
        """
        Apply mode-specific runtime parameters.
        """
        mode = str(mode).lower().strip()

        if mode not in ("video", "simulation", "live"):
            raise ValueError(f"Unsupported input mode: {mode}")

        self.input_mode = mode

        if mode == "video":
            self.crop_size = self.video_crop_size
            self.y_spacing = self.video_y_spacing
            self.fh_dy_mm_per_frame = self.video_fh_dy_mm_per_frame
            self.enable_stabilization = self.video_enable_stabilization
            self.stab_win = self.video_stab_win
            self.stab_search = self.video_stab_search
            self.stab_cc_thresh = self.video_stab_cc_thresh

        elif mode == "simulation":
            self.crop_size = self.sim_crop_size
            self.y_spacing = self.sim_y_spacing
            self.fh_dy_mm_per_frame = self.sim_fh_dy_mm_per_frame
            self.enable_stabilization = self.sim_enable_stabilization
            self.stab_win = self.sim_stab_win
            self.stab_search = self.sim_stab_search
            self.stab_cc_thresh = self.sim_stab_cc_thresh

        elif mode == "live":
            self.crop_size = self.live_crop_size
            self.y_spacing = self.live_y_spacing
            self.fh_dy_mm_per_frame = self.live_fh_dy_mm_per_frame
            self.enable_stabilization = self.live_enable_stabilization
            self.stab_win = self.live_stab_win
            self.stab_search = self.live_stab_search
            self.stab_cc_thresh = self.live_stab_cc_thresh
