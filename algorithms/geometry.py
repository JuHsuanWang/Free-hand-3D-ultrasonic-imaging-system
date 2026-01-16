# algorithms/geometry.py
import numpy as np


def img_to_world_xz(img_x: int, img_y: int, frame_h: int):
    """Image (x,y) -> world (X,Z) on XZ plane where Z is flipped."""
    return float(img_x), float(frame_h - img_y)


def world_to_img_xz(world_x: float, world_z: float, frame_h: int):
    """World (X,Z) -> image (x,y) with Z flipped back."""
    return int(world_x), int(frame_h - world_z)


def full_img_to_world_3d(img_x: int, img_y: int, frame_idx: int, frame_h: int, y_spacing: float):
    """ROI image (x,y) at frame index -> world (X,Y,Z)."""
    world_x = float(img_x)
    world_z = float(frame_h - img_y)
    world_y = float(frame_idx) * float(y_spacing)
    return world_x, world_y, world_z


def generate_grid_points(h: int, w: int, grid: int = 5, margin: int = 20) -> np.ndarray:
    """Generate grid×grid points in pixel coordinates (x,z)."""
    xs = np.linspace(margin, w - 1 - margin, grid)
    zs = np.linspace(margin, h - 1 - margin, grid)
    pts = []
    for zz in zs:
        for xx in xs:
            pts.append((int(round(xx)), int(round(zz))))
    return np.array(pts, dtype=np.int32)  # (N,2)
