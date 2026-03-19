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
    video_y_spacing: float = 5.4
    video_fh_dy_mm_per_frame: float = 0.1950
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
    save_stab_debug: bool = True

    # --- Out-of-plane rotation (beta/gamma) post-processing ---
    enable_beta_gamma_median_filter: bool = False
    beta_gamma_median_win: int = 5

    crop_top_frac: float = 0.25
    enable_crop_top: bool = True

    # --- Ground-truth (EM tracker) comparison plot ---
    enable_gt_plot: bool = False
    tracker_csv_path: str = "tracker_data_1.csv"
    tracker_port: str = "Port:11"
    gt_plot_filename: str = "gt_compare.png"

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