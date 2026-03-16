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
    # Video / live mode
    video_crop_size: int = 100
    video_fh_dy_mm_per_frame: float = 0.1

    # Simulation mode
    sim_crop_size: int = 50
    sim_fh_dy_mm_per_frame: float = 0.1

    # --------------------------------------------------
    # Segmentation settings
    # --------------------------------------------------
    bright_low: int = 140     # keep pixels >= bright_low
    bright_high: int = 220    # keep pixels <= bright_high
    min_contour_area: int = 2000

    # Performance
    num_workers: int = multiprocessing.cpu_count()

    # Output (will be overwritten to output/<run_name>/...)
    save_png: bool = True
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
    enable_gt_plot: bool = True
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

        if mode == "simulation":
            self.crop_size = self.sim_crop_size
            self.fh_dy_mm_per_frame = self.sim_fh_dy_mm_per_frame
        else:
            # both offline video and live capture use the same video defaults
            self.crop_size = self.video_crop_size
            self.fh_dy_mm_per_frame = self.video_fh_dy_mm_per_frame