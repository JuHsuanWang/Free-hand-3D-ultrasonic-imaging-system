# main.py
import os
import sys
from PyQt5 import QtWidgets, QtGui

from config import AppConfig
from core.loader import VideoLoader
from core.session import SessionState
from gui.window import MainWindow


def _make_run_name(left_path: str, right_path: str) -> str:
    """
    Build a stable run name from existing filenames.
    Recommended: use right video basename + optional left basename suffix.
    Example: R_20260104_abc + __L_20260104_def
    """
    l_base = os.path.splitext(os.path.basename(left_path))[0]
    r_base = os.path.splitext(os.path.basename(right_path))[0]

    # If they are simply "L" and "R", keep it clean.
    if l_base.lower() in ("l", "left") and r_base.lower() in ("r", "right"):
        return "LR"

    # Default: use R as primary, keep L for disambiguation
    return f"{r_base}__{l_base}"


def main():
    left_video = "Ln.avi"
    right_video = "Rn.avi"

    if len(sys.argv) >= 3:
        left_video = sys.argv[1]
        right_video = sys.argv[2]

    for v in (left_video, right_video):
        if not os.path.exists(v):
            print(f"Error: '{v}' not found")
            sys.exit(1)

    cfg = AppConfig()

    # --- NEW: output/<run_name>/... ---
    cfg.run_name = _make_run_name(left_video, right_video)
    run_dir = os.path.join(cfg.output_root, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Put everything under one clean folder
    cfg.png_out_dir = os.path.join(run_dir, "cropped_frames")
    cfg.stabilized_out_dir = os.path.join(run_dir, "stabilized_frames")
    # (Optional) if later you save pointcloud/surface, put them here too:
    # cfg.pointcloud_out_dir = os.path.join(run_dir, "pointcloud")
    # cfg.surface_out_dir = os.path.join(run_dir, "surface")

    print(f"[Output] {run_dir}")

    # Load frames (backend I/O)
    loader = VideoLoader(output_fps=cfg.output_fps)
    sess = SessionState(left_video_path=left_video, right_video_path=right_video)

    print("=" * 50)
    print("Video Crop & 3D Visualizer (Modular)")
    print("=" * 50)

    print("\nLoading videos...")
    sess.left_frames_original = loader.extract_frames(left_video)
    sess.right_frames_original = loader.extract_frames(right_video)
    sess.ensure_original_dims()

    print(f"Original frame: {sess.orig_frame_w}x{sess.orig_frame_h}")
    print(f"L: {len(sess.left_frames_original)}, R: {len(sess.right_frames_original)} frames")

    # Start Qt
    app = QtWidgets.QApplication(sys.argv)
    base_font = QtGui.QFont("Segoe UI", 12)
    app.setFont(base_font)

    win = MainWindow(session=sess, cfg=cfg)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
