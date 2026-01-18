# config.py
from dataclasses import dataclass
import multiprocessing


@dataclass
class AppConfig:
    """Centralized configuration for the whole app."""

    # Output root
    output_root: str = "output"
    run_name: str = "run"  # will be overwritten in main.py based on filename

    # Video sampling
    output_fps: int = 10

    # Crop settings
    crop_size: int = 100
    y_spacing: float = 5.3

    # Segmentation settings
    bright_low: int = 140     # keep pixels >= bright_low
    bright_high: int = 220    # keep pixels <= bright_high
    min_contour_area: int = 500


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

    # Point cloud / surface
    enable_surface: bool = False
    enable_point_cloud: bool = True
    voxel_size: float = 6.0
    max_points: int = 12000
    surface_alpha: float = 25.0
    surface_opacity: float = 0.35


