# gui/visualizer.py
from __future__ import annotations
from typing import Optional, List, Tuple
import os
import cv2
import numpy as np
import pyvista as pv
from concurrent.futures import ThreadPoolExecutor

from algorithms.geometry import (
    img_to_world_xz,
    world_to_img_xz,
    full_img_to_world_3d,
)
from core.session import SessionState
from config import AppConfig
from algorithms.stabilizer import StabilizerConfig, SequenceStabilizer, to_gray_u8
from algorithms.out_of_plane import OutOfPlaneConfig, compute_lr_heatmap_like_matlab



class VisualizerController:
    """
    Frontend controller for PyVista scene:
    - Handles click callbacks (ROI + crop center)
    - Draws ROI rectangle, red crop box, image planes, point cloud
    - Delegates data operations to SessionState and algorithms
    """

    def __init__(self, session: SessionState, cfg: AppConfig):
        self.sess = session
        self.cfg = cfg

        # PyVista plotter is injected by Qt window (pyvistaqt QtInteractor)
        self.plotter = None

        # Status & selection actors
        self.status_actor = None
        self.roi_actors: List = []
        self.selection_box_actor = None
        self.grid9_actor = None
        self.selection_point_actor = None
        self.band_actor = None

        # Rendered scene actors for toggles
        self.frame_actors: List = []
        self.point_cloud_actor = None
        self.surface_actor = None

        # Overlay actors in 3D
        self.crop_border_actors: List = []   # red boxes
        self.band_border_actors: List = []   # yellow bands

        # injected by MainWindow: QLabel on top of QtInteractor
        self.heatmap_overlay_label = None


    # -------------------------
    # UI helpers
    # -------------------------

    def update_status(self, text: str):
        if self.status_actor:
            try:
                self.plotter.remove_actor(self.status_actor)
            except Exception:
                pass
        self.status_actor = self.plotter.add_text(
            text,
            position="upper_left",
            font_size=14,
            color="black",
            font="times",
        )

    def _set_actor_visibility(self, actor, visible: bool):
        if actor is None:
            return
        try:
            actor.SetVisibility(1 if visible else 0)
        except Exception:
            try:
                actor.visibility = visible
            except Exception:
                pass

    def toggle_frames(self):
        self.sess.frames_visible = not self.sess.frames_visible
        for a in self.frame_actors:
            self._set_actor_visibility(a, self.sess.frames_visible)
        self.plotter.render()

    def toggle_point_cloud(self):
        self.sess.point_cloud_visible = not self.sess.point_cloud_visible
        self._set_actor_visibility(self.point_cloud_actor, self.sess.point_cloud_visible)
        self._set_actor_visibility(self.surface_actor, self.sess.point_cloud_visible)
        self.plotter.render()

    def toggle_crop_box(self, visible: bool):
        self.sess.crop_box_visible = bool(visible)
        for a in self.crop_border_actors:
            self._set_actor_visibility(a, self.sess.crop_box_visible)
        self.plotter.render()

    def toggle_band_box(self, visible: bool):
        self.sess.band_box_visible = bool(visible)
        for a in self.band_border_actors:
            self._set_actor_visibility(a, self.sess.band_box_visible)
        self.plotter.render()


    # -------------------------
    # Coordinate conversion
    # -------------------------

    def img_to_world(self, img_x: int, img_y: int, frame_h: int):
        return img_to_world_xz(img_x, img_y, frame_h)

    def world_to_img(self, world_x: float, world_z: float, frame_h: int):
        return world_to_img_xz(world_x, world_z, frame_h)

    # -------------------------
    # Phase 1: ROI selection
    # -------------------------

    def build_initial_scene(self):
        """Show the first frame for ROI selection."""
        p = self.plotter
        s = self.sess

        rgb = cv2.cvtColor(s.right_frames_original[0], cv2.COLOR_BGR2RGB)
        tex = pv.numpy_to_texture(rgb)

        pts = np.array([
            [0, 0, s.orig_frame_h],
            [s.orig_frame_w, 0, s.orig_frame_h],
            [s.orig_frame_w, 0, 0],
            [0, 0, 0],
        ], dtype=np.float32)
        faces = np.array([[4, 0, 1, 2, 3]])
        mesh = pv.PolyData(pts, faces)
        mesh.active_texture_coordinates = np.array(
            [[0, 1], [1, 1], [1, 0], [0, 0]],
            dtype=np.float32
        )
        p.add_mesh(mesh, texture=tex)

        # Camera aligned to XZ plane
        cx, cz = s.orig_frame_w / 2, s.orig_frame_h / 2
        dist = max(s.orig_frame_w, s.orig_frame_h) * 1.2
        p.camera.position = (cx, -dist, cz)
        p.camera.focal_point = (cx, 0, cz)
        p.camera.up = (0, 0, 1)

        self.update_status(
            "Step 1: Select ROI (ultrasound region)\n"
            "Click TOP-LEFT corner first\n"
            "Press 'r' to reset"
        )

        # Events
        p.track_click_position(callback=self.on_roi_click, side="left")
        p.add_key_event("Return", self.confirm_selection)  # Enter to confirm current phase
        p.add_key_event("r", self.reset_roi)

        p.render()

    def on_roi_click(self, point):
        """Handle clicks during ROI selection."""
        if self.sess.roi_confirmed:
            return

        world_x, world_z = float(point[0]), float(point[2])
        if world_x < 0 or world_x > self.sess.orig_frame_w or world_z < 0 or world_z > self.sess.orig_frame_h:
            return

        img_x, img_y = self.world_to_img(world_x, world_z, self.sess.orig_frame_h)
        img_x = max(0, min(self.sess.orig_frame_w - 1, img_x))
        img_y = max(0, min(self.sess.orig_frame_h - 1, img_y))

        if self.sess.roi_selection_phase == 1:
            self.sess.roi_pt1 = (img_x, img_y)
            self.sess.roi_selection_phase = 2
            self.update_roi_visuals()
            self.update_status(
                f"Top-left: ({img_x}, {img_y})\n"
                "Now click BOTTOM-RIGHT corner\n"
                "Press 'r' to reset"
            )
        else:
            self.sess.roi_pt2 = (img_x, img_y)
            self.update_roi_visuals()
            self.update_status(
                f"Top-left: {self.sess.roi_pt1}\n"
                f"Bottom-right: ({img_x}, {img_y})\n"
                "Press 'Enter' to confirm | 'r' to reset"
            )

        self.plotter.render()

    def update_roi_visuals(self):
        """Draw ROI points and rectangle on the current scene."""
        for actor in self.roi_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.roi_actors = []

        if self.sess.roi_pt1:
            wx, wz = self.img_to_world(self.sess.roi_pt1[0], self.sess.roi_pt1[1], self.sess.orig_frame_h)
            sphere1 = pv.Sphere(radius=10, center=(wx, 0.5, wz))
            self.roi_actors.append(self.plotter.add_mesh(sphere1, color="cyan"))

        if self.sess.roi_pt2:
            wx, wz = self.img_to_world(self.sess.roi_pt2[0], self.sess.roi_pt2[1], self.sess.orig_frame_h)
            sphere2 = pv.Sphere(radius=10, center=(wx, 0.5, wz))
            self.roi_actors.append(self.plotter.add_mesh(sphere2, color="cyan"))

        if self.sess.roi_pt1 and self.sess.roi_pt2:
            x1, y1 = self.sess.roi_pt1
            x2, y2 = self.sess.roi_pt2
            wx1, wz1 = self.img_to_world(x1, y1, self.sess.orig_frame_h)
            wx2, wz2 = self.img_to_world(x2, y2, self.sess.orig_frame_h)

            wxmin, wxmax = min(wx1, wx2), max(wx1, wx2)
            wzmin, wzmax = min(wz1, wz2), max(wz1, wz2)

            lines = [
                pv.Line((wxmin, 0.5, wzmin), (wxmax, 0.5, wzmin)),
                pv.Line((wxmax, 0.5, wzmin), (wxmax, 0.5, wzmax)),
                pv.Line((wxmax, 0.5, wzmax), (wxmin, 0.5, wzmax)),
                pv.Line((wxmin, 0.5, wzmax), (wxmin, 0.5, wzmin)),
            ]
            box_mesh = lines[0]
            for line in lines[1:]:
                box_mesh = box_mesh.merge(line)

            self.roi_actors.append(
                self.plotter.add_mesh(box_mesh, color="cyan", line_width=4, render_lines_as_tubes=True)
            )

    def reset_roi(self):
        """Reset ROI selection state."""
        self.sess.roi_pt1 = None
        self.sess.roi_pt2 = None
        self.sess.roi_selection_phase = 1

        for actor in self.roi_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.roi_actors = []

        self.update_status(
            "ROI Reset\n"
            "Click TOP-LEFT corner of ultrasound region"
        )
        self.plotter.render()

    def confirm_roi(self):
        """Confirm ROI and switch to crop-center selection scene."""
        if self.sess.roi_pt1 is None or self.sess.roi_pt2 is None:
            self.update_status("Please select both corners first!")
            self.plotter.render()
            return

        if self.sess.roi_confirmed:
            return

        self.sess.roi_confirmed = True

        # Apply ROI crop to frames
        self.apply_roi_crop()

        # Clear scene and show ROI-cropped frame for next phase
        self.plotter.clear_actors()
        self.show_cropped_frame_for_selection()

    def apply_roi_crop(self):
        """Apply ROI crop to all frames (in-memory)."""
        x1, y1 = self.sess.roi_pt1
        x2, y2 = self.sess.roi_pt2

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        self.sess.left_frames = self.sess.left_frames_original[:, y1:y2, x1:x2, :].copy()
        self.sess.right_frames = self.sess.right_frames_original[:, y1:y2, x1:x2, :].copy()

        self.sess.ensure_roi_dims()
        print(f"  ROI cropped to: {self.sess.frame_w}x{self.sess.frame_h}")

    # -------------------------
    # Phase 2: Crop center selection
    # -------------------------

    def show_cropped_frame_for_selection(self):
        """Show the first ROI-cropped frame to select crop center (red point/box)."""
        rgb = cv2.cvtColor(self.sess.right_frames[0], cv2.COLOR_BGR2RGB)
        tex = pv.numpy_to_texture(rgb)

        pts = np.array([
            [0, 0, self.sess.frame_h],
            [self.sess.frame_w, 0, self.sess.frame_h],
            [self.sess.frame_w, 0, 0],
            [0, 0, 0],
        ], dtype=np.float32)
        faces = np.array([[4, 0, 1, 2, 3]])
        mesh = pv.PolyData(pts, faces)
        mesh.active_texture_coordinates = np.array(
            [[0, 1], [1, 1], [1, 0], [0, 0]],
            dtype=np.float32
        )
        self.plotter.add_mesh(mesh, texture=tex)

        # Camera
        cx, cz = self.sess.frame_w / 2, self.sess.frame_h / 2
        dist = max(self.sess.frame_w, self.sess.frame_h) * 1.2
        self.plotter.camera.position = (cx, -dist, cz)
        self.plotter.camera.focal_point = (cx, 0, cz)
        self.plotter.camera.up = (0, 0, 1)

        self.update_status(
            "Step 2: Select crop center (100x100)\n"
            "Click to select | 'Enter' to confirm | 'q' to quit"
        )

        self.plotter.track_click_position(callback=self.on_crop_click, side="left")
        self.plotter.add_key_event("q", lambda: self._quit())
        self.plotter.render()

    def _quit(self):
        """Quit handler (safe close)."""
        try:
            self.plotter.close()
        except Exception:
            pass

    def on_crop_click(self, point):
        """Handle clicks during crop center selection (center cell of a 3x3 grid)."""
        if self.sess.selection_confirmed:
            return

        world_x, world_z = float(point[0]), float(point[2])
        if world_x < 0 or world_x > self.sess.frame_w or world_z < 0 or world_z > self.sess.frame_h:
            return

        img_x, img_y = self.world_to_img(world_x, world_z, self.sess.frame_h)
        img_x = max(0, min(self.sess.frame_w - 1, img_x))
        img_y = max(0, min(self.sess.frame_h - 1, img_y))

        # --- NEW: clamp so the whole 3x3 grid stays inside ROI image ---
        cell = int(self.cfg.crop_size)             # 100
        half_grid = (3 * cell) // 2                # 150 when cell=100
        # valid center range: [half_grid, (w-1)-half_grid]
        x_min = half_grid
        x_max = (self.sess.frame_w - 1) - half_grid
        y_min = half_grid
        y_max = (self.sess.frame_h - 1) - half_grid

        # If ROI is too small to fit 3x3, fallback to original behavior (still show red box only)
        if x_min <= x_max and y_min <= y_max:
            img_x = int(max(x_min, min(x_max, img_x)))
            img_y = int(max(y_min, min(y_max, img_y)))
        # --------------------------------------------------------------

        self.sess.click_point = (img_x, img_y)
        # Update depth(mm) and LR-distance from the selected center point.
        # Depth uses ROI vertical span (roi_depth_mm) and assumes y=0 is shallow.
        self.sess.update_click_metrics(img_y)
        self.sess.click_depth_mm = float(img_y) * float(self.sess.depth_mm_per_px)
        self.update_crop_visuals()

        self.update_status(
            f"Crop center (middle of 3x3 grid): ({img_x}, {img_y})\n"
            f"Depth: {self.sess.click_depth_mm:.2f} mm | LR: {self.sess.click_lr_distance:.4f}\n"
            "Press 'Enter' to confirm | 'q' to quit"
        )
        self.plotter.render()

    def update_crop_visuals(self):
        """Draw red center point, central 100x100 red box, and a 3x3 grid (each cell 100x100)."""
        if self.sess.click_point is None:
            return

        # remove old actors
        if self.selection_point_actor:
            try:
                self.plotter.remove_actor(self.selection_point_actor)
            except Exception:
                pass
        if self.selection_box_actor:
            try:
                self.plotter.remove_actor(self.selection_box_actor)
            except Exception:
                pass
        if self.grid9_actor:
            try:
                self.plotter.remove_actor(self.grid9_actor)
            except Exception:
                pass
            self.grid9_actor = None

        if self.band_actor:
            try:
                self.plotter.remove_actor(self.band_actor)
            except Exception:
                pass
            self.band_actor = None

        img_x, img_y = self.sess.click_point
        world_x, world_z = self.img_to_world(img_x, img_y, self.sess.frame_h)

        cell = int(self.cfg.crop_size)       # 100
        half_cell = cell // 2                # 50
        half_grid = (3 * cell) // 2          # 150

        # ---- red center point ----
        center_sphere = pv.Sphere(radius=10, center=(world_x, 0.5, world_z))
        self.selection_point_actor = self.plotter.add_mesh(center_sphere, color="red")

        # ---- central red box (middle cell) ----
        box_x1, box_x2 = world_x - half_cell, world_x + half_cell
        box_z1, box_z2 = world_z - half_cell, world_z + half_cell

        lines = [
            pv.Line((box_x1, 0.5, box_z1), (box_x2, 0.5, box_z1)),
            pv.Line((box_x2, 0.5, box_z1), (box_x2, 0.5, box_z2)),
            pv.Line((box_x2, 0.5, box_z2), (box_x1, 0.5, box_z2)),
            pv.Line((box_x1, 0.5, box_z2), (box_x1, 0.5, box_z1)),
        ]
        box_mesh = lines[0]
        for line in lines[1:]:
            box_mesh = box_mesh.merge(line)

        self.selection_box_actor = self.plotter.add_mesh(
            box_mesh, color="red", line_width=5, render_lines_as_tubes=True
        )

        # ---- NEW (REPLACE): true 3x3 grid with uniform line style ----
        cell = int(self.cfg.crop_size)      # 100
        half_cell = cell // 2              # 50

        # 4 boundaries per axis for a 3x3 grid:
        # [-150, -50, +50, +150] relative to center when cell=100
        offsets = [-1.5 * cell, -0.5 * cell, 0.5 * cell, 1.5 * cell]

        grid_lines = []

        # vertical lines (x fixed, z spans full grid height)
        for dx in offsets:
            x = world_x + dx
            z1 = world_z + offsets[0]
            z2 = world_z + offsets[-1]
            grid_lines.append(pv.Line((x, 0.5, z1), (x, 0.5, z2)))

        # horizontal lines (z fixed, x spans full grid width)
        for dz in offsets:
            z = world_z + dz
            x1 = world_x + offsets[0]
            x2 = world_x + offsets[-1]
            grid_lines.append(pv.Line((x1, 0.5, z), (x2, 0.5, z)))

        grid_mesh = grid_lines[0]
        for ln in grid_lines[1:]:
            grid_mesh = grid_mesh.merge(ln)

        # uniform style: outer border is NOT special; same thickness everywhere
        self.grid9_actor = self.plotter.add_mesh(
            grid_mesh,
            color="orange",
            line_width=3,
            render_lines_as_tubes=True
        )
        # -------------------------------------------------------------
        # ---- Yellow band = middle row (4-5-6) of 3x3 grid ----
        cell = int(self.cfg.crop_size)

        band_x_left  = world_x - 1.5 * cell
        band_x_right = world_x + 1.5 * cell
        band_z_top   = world_z + 0.5 * cell
        band_z_bot   = world_z - 0.5 * cell

        band_lines = [
            pv.Line((band_x_left, 0.5, band_z_top), (band_x_right, 0.5, band_z_top)),
            pv.Line((band_x_right, 0.5, band_z_top), (band_x_right, 0.5, band_z_bot)),
            pv.Line((band_x_right, 0.5, band_z_bot), (band_x_left, 0.5, band_z_bot)),
            pv.Line((band_x_left, 0.5, band_z_bot), (band_x_left, 0.5, band_z_top)),
        ]
        band_mesh = band_lines[0]
        for ln in band_lines[1:]:
            band_mesh = band_mesh.merge(ln)

        self.band_actor = self.plotter.add_mesh(
            band_mesh,
            color="yellow",
            line_width=4,
            render_lines_as_tubes=True
        )
        # --------------------------------------------------------------

    # -------------------------
    # Confirm selection (Enter)
    # -------------------------

    def confirm_selection(self):
        """Enter key event: confirm ROI first, then confirm crop center and process."""
        if not self.sess.roi_confirmed:
            self.confirm_roi()
            return

        if self.sess.click_point is None:
            self.update_status("Please select crop center first!")
            self.plotter.render()
            return

        if self.sess.selection_confirmed:
            return

        self.sess.selection_confirmed = True
        self.plotter.clear_actors()

        self.run_processing()
        self.build_3d_view()

    # -------------------------
    # Backend processing
    # -------------------------

    def crop_frames_vectorized(self, frames: np.ndarray, center_xy: Tuple[int, int]) -> np.ndarray:
        """Crop (N,H,W,C) around center to (N,crop,crop,C) with zero-padding if needed."""
        if len(frames) == 0:
            return np.array([])

        x, y = center_xy
        half = self.cfg.crop_size // 2
       
        n, h, w, c = frames.shape

        x1, x2 = max(0, x - half), min(w, x + half)
        y1, y2 = max(0, y - half), min(h, y + half)

        cropped = frames[:, y1:y2, x1:x2, :]
        ah, aw = cropped.shape[1:3]

        if aw < self.cfg.crop_size or ah < self.cfg.crop_size:
            padded = np.zeros((n, self.cfg.crop_size, self.cfg.crop_size, c), dtype=np.uint8)
            px, py = (self.cfg.crop_size - aw) // 2, (self.cfg.crop_size - ah) // 2
            padded[:, py:py + ah, px:px + aw, :] = cropped
            return padded

        return cropped
        
    def crop_band_full_width(self, frames: np.ndarray, center_y: int) -> np.ndarray:
        """
        Crop a full-width horizontal band:
        output shape: (N, crop_size, frame_w, C)
        band is centered at center_y, height=crop_size, width=full frame_w.
        Pads with zeros if near top/bottom.
        """
        if len(frames) == 0:
            return np.array([])

        band_h = 50  # use 50 as band height
        half = band_h // 2

        n, h, w, c = frames.shape
        y1 = max(0, center_y - half)
        y2 = min(h, center_y + half)

        band = frames[:, y1:y2, 0:w, :]
        ah = band.shape[1]

        if ah < band_h:
            padded = np.zeros((n, band_h, w, c), dtype=np.uint8)
            py = (band_h - ah) // 2
            padded[:, py:py + ah, :, :] = band
            return padded

        return band

    def otsu_segmentation(self, image: np.ndarray, threshold: int) -> np.ndarray:
        """Detect dark regions using fixed threshold and contour area filtering."""
        if image.ndim == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return np.empty((0, 2), dtype=np.float32)

        all_points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.cfg.min_contour_area:
                points = contour.reshape(-1, 2)
                all_points.extend(points.tolist())

        if not all_points:
            return np.empty((0, 2), dtype=np.float32)
        return np.array(all_points, dtype=np.float32)

    def process_all_segmentations(self):
        """Run segmentation for all ROI frames (multi-threaded)."""
        n = len(self.sess.right_frames)
        if n == 0:
            self.sess.contour_points_list = []
            return

        self.sess.contour_points_list = [None] * n

        def _seg_one(i: int):
            try:
                pts = self.otsu_segmentation(self.sess.right_frames[i], self.cfg.dark_threshold)
                return i, pts
            except Exception:
                return i, np.empty((0, 2), dtype=np.float32)

        with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as ex:
            for i, pts in ex.map(_seg_one, range(n)):
                self.sess.contour_points_list[i] = pts

        total_points = sum(len(p) for p in self.sess.contour_points_list if p is not None)
        print(f"  Segmentation complete (threshold={self.cfg.dark_threshold}): {total_points} total contour points")

    def run_processing(self):
        """Full pipeline after selection: stabilization -> crop -> segmentation -> optional save."""
        n = min(len(self.sess.left_frames), len(self.sess.right_frames))
        if n == 0:
            raise ValueError("No frames after ROI crop.")

        self.update_status("Processing...")
        self.plotter.render()

        # Stabilize full ROI (optional)
        if self.cfg.enable_stabilization:
            self.update_status("Stabilizing Full ROI...")
            self.plotter.render()

            # --- NEW: debug output dir (match third-code naming) ---
            # put stabilized_frames next to png_out_dir, e.g.
            # output/cropped_frames  -> output/stabilized_frames/R
            base_dir = os.path.dirname(self.cfg.png_out_dir.rstrip("/\\"))
            if base_dir == "":
                base_dir = "."
            debug_r_dir = os.path.join(base_dir, "stabilized_frames", "R")
            # ------------------------------------------------------

            stab_cfg = StabilizerConfig(
                crop_size=self.cfg.crop_size,
                stab_grid=self.cfg.stab_grid,
                stab_win=self.cfg.stab_win,
                stab_search=self.cfg.stab_search,
                stab_cc_thresh=self.cfg.stab_cc_thresh,
                dx_mm=self.cfg.dx_mm,
                dz_mm=self.cfg.dz_mm,

                # --- NEW: enable debug images like third code ---
                save_debug=True,
                debug_out_dir=debug_r_dir,
                debug_prefix="R",
                # ----------------------------------------------
            )
            stabilizer = SequenceStabilizer(stab_cfg)
            self.sess.right_frames, self.sess.left_frames = stabilizer.stabilize_full_roi_inplace(
                self.sess.right_frames, self.sess.left_frames, self.sess.click_point
            )

        # Refresh 100x100 crops based on possibly stabilized frames
        self.sess.cropped_left = self.crop_frames_vectorized(self.sess.left_frames[:n], self.sess.click_point)
        self.sess.cropped_right = self.crop_frames_vectorized(self.sess.right_frames[:n], self.sess.click_point)
        # NEW: full-width band crops for LK Y-heatmap
        _, cy = self.sess.click_point
        self.sess.band_left = self.crop_band_full_width(self.sess.left_frames[:n], cy)
        self.sess.band_right = self.crop_band_full_width(self.sess.right_frames[:n], cy)

        # Segmentation on ROI frames
        self.update_status("Running segmentation...")
        self.plotter.render()
        self.process_all_segmentations()

        # Save cropped outputs if enabled
        if self.cfg.save_png:
            out_dir = self.cfg.png_out_dir
            left_dir = os.path.join(out_dir, "L")
            right_dir = os.path.join(out_dir, "R")
            os.makedirs(left_dir, exist_ok=True)
            os.makedirs(right_dir, exist_ok=True)

            def _save_one(i: int):
                cv2.imwrite(os.path.join(left_dir, f"L_{i:04d}.png"), self.sess.cropped_left[i])
                cv2.imwrite(os.path.join(right_dir, f"R_{i:04d}.png"), self.sess.cropped_right[i])

            with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as ex:
                list(ex.map(_save_one, range(n)))

            print(f"  Saved crops to {out_dir}")

        self.update_status("Processing done.")
        self.plotter.render()

    def compute_y_heatmap_like_matlab(self):
        """
        MATLAB-like Y heatmap:
        - normalizeImage (brightness+contrast)
        - grid LK + valid_mask
        - r starts from l+1
        """
        if self.sess.band_left is None or self.sess.band_right is None:
            self.sess.y_heatmap = None
            self.sess.y_best_pairs = []
            return

        of_cfg = OutOfPlaneConfig(
            normalize_brightness=False,
            normalize_contrast=False,
            target_mean=128.0,
            target_std=50.0,
            grid_spacing=10,
            window_size=25,
            max_displacement=150.0,
            det_thresh=1e-6,
        )

        # Use full-width band (MATLAB-style strip)
        H, best = compute_lr_heatmap_like_matlab(self.sess.band_left, self.sess.band_right, of_cfg)
        self.sess.y_heatmap = H
        self.sess.y_best_pairs = best

    
    def show_out_of_plane_heatmap_overlay(self):
        """
        Render heatmap to PNG and show it in the overlay QLabel at the center of PyVista view.
        Click overlay to hide.
        """
        if self.heatmap_overlay_label is None:
            print("[YHeatmap] overlay label not set (did you assign in MainWindow?)")
            return

        if self.sess.y_heatmap is None or self.sess.y_heatmap.size == 0:
            return

        from PyQt5 import QtGui, QtCore, QtWidgets
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO

        H = self.sess.y_heatmap  # (nL, nR)

        fig = plt.figure(figsize=(7.2, 6.0), dpi=150)
        ax = fig.add_subplot(111)

        Ht = H.T  # (nR, nL)
        m = np.ma.masked_invalid(Ht)

        ax.set_facecolor("white")
        im = ax.imshow(m, cmap="gray", origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xlabel("Left RX Reference Frame (frame #)")
        ax.set_ylabel("Right RX Frame (frame #)")
        ax.set_title("Optical Flow Displacement Heat Map (L vs R)")
        # NEW: clamp Y axis to actual max frame (remove top whitespace)
        if self.sess.y_heatmap is not None and self.sess.y_heatmap.size > 0:
            ymax = int(self.sess.y_heatmap.shape[0] - 1)
            ax.set_ylim(0, ymax)
            ax.margins(y=0)

        if self.sess.y_best_pairs:
            # -----------------------------
            # 1. collect points
            # -----------------------------
            xs = np.array([l for (l, r, v) in self.sess.y_best_pairs], dtype=np.float64)
            ys = np.array([r for (l, r, v) in self.sess.y_best_pairs], dtype=np.float64)

            if len(xs) >= 5:
                # -----------------------------
                # 2. initial linear fit
                # -----------------------------
                a0, b0 = np.polyfit(xs, ys, 1)
                y_pred0 = a0 * xs + b0
                residuals = ys - y_pred0

                # -----------------------------
                # 3. MAD-based outlier removal
                # -----------------------------
                med = np.median(residuals)
                mad = np.median(np.abs(residuals - med)) + 1e-6
                inlier_mask = np.abs(residuals - med) < 2.0 * mad

                xs_in = xs[inlier_mask]
                ys_in = ys[inlier_mask]
                xs_out = xs[~inlier_mask]
                ys_out = ys[~inlier_mask]

                # -----------------------------
                # 4. final regression (inliers)
                # -----------------------------
                a, b = np.polyfit(xs_in, ys_in, 1)
                y_fit = a * xs_in + b

                # R^2
                ss_res = np.sum((ys_in - y_fit) ** 2)
                ss_tot = np.sum((ys_in - np.mean(ys_in)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                # -----------------------------
                # 5. draw points
                # -----------------------------
                # outliers (white)
                if len(xs_out) > 0:
                    ax.scatter(xs_out, ys_out, s=3, c="white", edgecolors="none", alpha=0.9)

                # inliers (red)
                ax.scatter(xs_in, ys_in, s=2, c="red")

                # -----------------------------
                # 6. draw regression line
                # -----------------------------
                x_line = np.array([0, np.max(xs)])
                y_line = a * x_line + b
                ax.plot(x_line, y_line, color="yellow", linewidth=1.5)

                # -----------------------------
                # 7. annotate R^2 and intercept
                # -----------------------------
                ax.text(
                    0.98, 0.02,
                    f"$R^2$ = {r2:.3f}\nIntercept (x=0): {b:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
                )

            else:
                # fallback: too few points
                ax.scatter(xs, ys, s=2, c="red")


        ax.grid(True, alpha=0.25)

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        pix = QtGui.QPixmap()
        pix.loadFromData(buf.getvalue(), "PNG")

        # scale to ~80% of vtk widget size (keep aspect)
        parent = self.heatmap_overlay_label.parentWidget()
        if parent is None:
            parent = self.heatmap_overlay_label
        max_w = int(parent.width() * 0.80)
        max_h = int(parent.height() * 0.80)
        pix = pix.scaled(max_w, max_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        self.heatmap_overlay_label.setPixmap(pix)
        self.heatmap_overlay_label.setFixedSize(pix.size())
        self.heatmap_overlay_label.setVisible(True)
        self.heatmap_overlay_label.raise_()

        # click-to-hide (install once)
        if not hasattr(self.heatmap_overlay_label, "_hide_installed"):
            self.heatmap_overlay_label._hide_installed = True

            def _mousePressEvent(evt):
                self.heatmap_overlay_label.setVisible(False)

            self.heatmap_overlay_label.mousePressEvent = _mousePressEvent


    def on_show_y_heatmap(self):
        if not self.sess.selection_confirmed:
            return

        # NEW: toggle behavior
        if self.heatmap_overlay_label.isVisible():
            self.heatmap_overlay_label.setVisible(False)
            return

        if self.sess.y_heatmap is None or self.sess.y_heatmap.size == 0:
            self.update_status("Computing out-of-plane (Y) heatmap (MATLAB-like)...")
            self.plotter.render()
            self.compute_y_heatmap_like_matlab()
            self.update_status("Ready.")
            self.plotter.render()

        self.show_out_of_plane_heatmap_overlay()

    # -------------------------
    # 3D Visualization
    # -------------------------

    def downsample_point_cloud(self, points_xyz: np.ndarray) -> np.ndarray:
        """Downsample point cloud using PyVista clean + random cap."""
        if points_xyz is None or len(points_xyz) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        cloud = pv.PolyData(np.asarray(points_xyz, dtype=np.float32))

        try:
            cloud_ds = cloud.clean(tolerance=float(self.cfg.voxel_size))
        except Exception:
            cloud_ds = cloud

        pts_ds = np.asarray(cloud_ds.points, dtype=np.float32)
        if len(pts_ds) <= self.cfg.max_points:
            return pts_ds

        idx = np.random.choice(len(pts_ds), size=self.cfg.max_points, replace=False)
        return pts_ds[idx]

    def reconstruct_surface(self, points_xyz: np.ndarray) -> Optional[pv.PolyData]:
        """Surface reconstruction with Delaunay 3D alpha shape."""
        if points_xyz is None or len(points_xyz) < 50:
            return None

        cloud = pv.PolyData(np.asarray(points_xyz, dtype=np.float32))
        try:
            vol = cloud.delaunay_3d(alpha=float(self.cfg.surface_alpha))
            surf = vol.extract_geometry()
            return surf
        except Exception as e:
            print(f"Surface reconstruction failed: {e}")
            return None

    def build_3d_view(self):
        """Render stacked image planes + crop planes + contour point cloud/surface."""
        s = self.sess
        n = len(s.right_frames)

        img_x, img_y = s.click_point
        crop_world_x, crop_world_z = self.img_to_world(img_x, img_y, s.frame_h)
        half = self.cfg.crop_size // 2
        
        self.update_status(f"Preparing {n} frames...")
        self.plotter.render()

        all_meshes = []
        all_contour_points = []

        # Render every 5 frames (same as your original)
        for i in range(0, n, 5):
            y_pos = i * self.cfg.y_spacing
            # --- NEW: yellow band border (4-5-6 row) for THIS frame y_pos ---
            cell = int(self.cfg.crop_size)
            band_z_top = crop_world_z + 0.5 * cell
            band_z_bot = crop_world_z - 0.5 * cell
            band_x_left = crop_world_x - 1.5 * cell
            band_x_right = crop_world_x + 1.5 * cell

            band_border_pts = np.array([
                [band_x_left,  y_pos, band_z_top],
                [band_x_right, y_pos, band_z_top],
                [band_x_right, y_pos, band_z_bot],
                [band_x_left,  y_pos, band_z_bot],
                [band_x_left,  y_pos, band_z_top],
            ], dtype=np.float32)
            band_border_mesh = pv.lines_from_points(band_border_pts)
            # --------------------------------------------------------------

            frame = s.right_frames[i]
            if frame.ndim == 3 and frame.shape[2] == 4:
                full_rgba = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            else:
                full_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_tex = pv.numpy_to_texture(full_rgba)

            full_pts = np.array([
                [0, y_pos, s.frame_h],
                [s.frame_w, y_pos, s.frame_h],
                [s.frame_w, y_pos, 0],
                [0, y_pos, 0],
            ], dtype=np.float32)
            full_faces = np.array([[4, 0, 1, 2, 3]])
            full_mesh = pv.PolyData(full_pts, full_faces)
            full_mesh.active_texture_coordinates = np.array(
                [[0, 1], [1, 1], [1, 0], [0, 0]],
                dtype=np.float32
            )

            crop_frame = s.cropped_right[i]
            if crop_frame.ndim == 3 and crop_frame.shape[2] == 4:
                crop_rgba = cv2.cvtColor(crop_frame, cv2.COLOR_BGRA2RGBA)
            else:
                crop_rgba = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            crop_tex = pv.numpy_to_texture(crop_rgba)

            crop_z_top = crop_world_z + half
            crop_z_bot = crop_world_z - half
            crop_x_left = crop_world_x - half
            crop_x_right = crop_world_x + half

            crop_pts = np.array([
                [crop_x_left, y_pos, crop_z_top],
                [crop_x_right, y_pos, crop_z_top],
                [crop_x_right, y_pos, crop_z_bot],
                [crop_x_left, y_pos, crop_z_bot],
            ], dtype=np.float32)
            crop_faces = np.array([[4, 0, 1, 2, 3]])
            crop_mesh = pv.PolyData(crop_pts, crop_faces)
            crop_mesh.active_texture_coordinates = np.array(
                [[0, 1], [1, 1], [1, 0], [0, 0]],
                dtype=np.float32
            )

            border_pts = np.array([
                [crop_x_left, y_pos, crop_z_top],
                [crop_x_right, y_pos, crop_z_top],
                [crop_x_right, y_pos, crop_z_bot],
                [crop_x_left, y_pos, crop_z_bot],
                [crop_x_left, y_pos, crop_z_top],
            ], dtype=np.float32)
            border_mesh = pv.lines_from_points(border_pts)

            all_meshes.append({
                "full": (full_mesh, full_tex),
                "crop": (crop_mesh, crop_tex),
                "crop_border": border_mesh,
                "band_border": band_border_mesh
            })

            contour_pts_2d = s.contour_points_list[i]
            if contour_pts_2d is not None and len(contour_pts_2d) > 0:
                for pt in contour_pts_2d:
                    wx, wy, wz = full_img_to_world_3d(int(pt[0]), int(pt[1]), i, s.frame_h, self.cfg.y_spacing)
                    all_contour_points.append([wx, wy, wz])

        self.update_status(f"Rendering {n} frames...")
        self.plotter.render()

        # Render planes
        self.frame_actors = []
        self.crop_border_actors = []
        self.band_border_actors = []
        for data in all_meshes:
            a_full = self.plotter.add_mesh(data["full"][0], texture=data["full"][1], opacity=0.15)
            a_crop = self.plotter.add_mesh(data["crop"][0], texture=data["crop"][1], opacity=0.95)

            # red crop box
            a_crop_border = self.plotter.add_mesh(
                data["crop_border"],
                color="red",
                line_width=2
            )
            self._set_actor_visibility(a_crop_border, self.sess.crop_box_visible)
            self.crop_border_actors.append(a_crop_border)

            # yellow band box
            a_band_border = self.plotter.add_mesh(
                data["band_border"],
                color="yellow",
                line_width=2
            )
            self._set_actor_visibility(a_band_border, self.sess.band_box_visible)
            self.band_border_actors.append(a_band_border)

            self.frame_actors.extend([a_full, a_crop])

        # Render point cloud + surface
        self.point_cloud_actor = None
        self.surface_actor = None

        if len(all_contour_points) > 0:
            raw_pts = np.asarray(all_contour_points, dtype=np.float32)
            ds_pts = self.downsample_point_cloud(raw_pts)
            print(f"  Point cloud: raw={len(raw_pts)}, downsampled={len(ds_pts)}")

            if self.cfg.enable_point_cloud and len(ds_pts) > 0:
                cloud = pv.PolyData(ds_pts)
                self.point_cloud_actor = self.plotter.add_mesh(
                    cloud,
                    color="yellow",
                    point_size=3,
                    render_points_as_spheres=True,
                    style="points"
                )

            if self.cfg.enable_surface and len(ds_pts) > 50:
                surf = self.reconstruct_surface(ds_pts)
                if surf is not None and surf.n_points > 0:
                    self.surface_actor = self.plotter.add_mesh(
                        surf,
                        opacity=float(self.cfg.surface_opacity),
                        smooth_shading=True
                    )

        self.plotter.add_axes(xlabel="X", ylabel="Y (Frame)", zlabel="Z")
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()

        self.update_status(
            f"Done! {n} frames, {len(all_contour_points)} contour points\n"
            "Drag: Rotate | Right-drag: Pan | Scroll: Zoom | 'q': Quit"
        )
        self.plotter.render()
