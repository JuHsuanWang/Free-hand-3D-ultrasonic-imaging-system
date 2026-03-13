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
    cfg = AppConfig()

    # Generate a default run name since no video is loaded yet
    import datetime
    cfg.run_name = f"Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.output_root, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Set output directories
    cfg.png_out_dir = os.path.join(run_dir, "cropped_frames")
    cfg.stabilized_out_dir = os.path.join(run_dir, "stabilized_frames")

    print(f"[Output] {run_dir}")

    # Initialize an empty SessionState. Videos will be loaded via GUI.
    sess = SessionState(left_video_path="", right_video_path="")

    print("=" * 50)
    print("Freehand 3D US System")
    print("=" * 50)
    print("Ready. Waiting for user to select data source...")

    # Start Qt
    app = QtWidgets.QApplication(sys.argv)
    base_font = QtGui.QFont("Segoe UI", 12)
    app.setFont(base_font)

    win = MainWindow(session=sess, cfg=cfg)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
