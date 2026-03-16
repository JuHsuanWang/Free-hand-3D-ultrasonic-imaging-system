# gui/window.py
from __future__ import annotations
from PyQt5 import QtWidgets, QtGui, QtCore
from pyvistaqt import QtInteractor

from core.session import SessionState
from config import AppConfig
from gui.visualizer import VisualizerController
import os
import subprocess
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from core.loader import VideoLoader, ImageSequenceLoader

class CaptureThread(QtCore.QThread):
    """
    Background thread to run the PSRT real-time capture script.
    This prevents the main GUI from freezing during data acquisition.
    """
    finished_signal = QtCore.pyqtSignal(bool, str)

    def run(self):
        try:
            # Name of your PSRT script
            script_name = "PSRT_Python - 茹瑄implementation.py"
            
            # Execute the script as a separate process
            result = subprocess.run(["python", script_name], capture_output=True, text=True)
            
            # Check if the process exited successfully
            if result.returncode == 0:
                self.finished_signal.emit(True, "Capture successful.")
            else:
                self.finished_signal.emit(False, result.stderr)
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class MainWindow(QtWidgets.QMainWindow):
    """PyQt main window: embeds PyVista interactor and provides control buttons."""

    def __init__(self, session: SessionState, cfg: AppConfig):
        super().__init__()
        self.sess = session
        self.cfg = cfg

        self.setWindowTitle("Freehand 3D US System")
        self.resize(1200, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: PyVista Qt interactor + overlay label (for heatmap)
        self.vtk_container = QtWidgets.QWidget(central)
        self.vtk_container.setObjectName("vtk_container")
        container_layout = QtWidgets.QGridLayout(self.vtk_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        self.vtk_widget = QtInteractor(self.vtk_container)
        container_layout.addWidget(self.vtk_widget, 0, 0)

        # Overlay label: centered on top of vtk_widget
        self.heatmap_overlay = QtWidgets.QLabel(self.vtk_container)
        self.heatmap_overlay.setObjectName("heatmap_overlay")
        self.heatmap_overlay.setAlignment(QtCore.Qt.AlignCenter)
        self.heatmap_overlay.setVisible(False)
        self.heatmap_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.heatmap_overlay.setStyleSheet(
            "QLabel#heatmap_overlay {"
            "  background-color: rgba(255,255,255,220);"
            "  border: 2px solid #444;"
            "  border-radius: 8px;"
            "}"
        )
        container_layout.addWidget(self.heatmap_overlay, 0, 0, alignment=QtCore.Qt.AlignCenter)

        layout.addWidget(self.vtk_container, stretch=1)


        # Right: control panel
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        # --- Data Source Group ---
        grp_source = QtWidgets.QGroupBox("1. Data Source")
        l_source = QtWidgets.QVBoxLayout()
        
        # Button for offline video mode
        self.btn_load_offline = QtWidgets.QPushButton("Load Offline Videos")
        self.btn_load_offline.setStyleSheet("background-color: #ADD8E6; padding: 10px; font-weight: bold;")

        # Button for simulation mode
        self.btn_load_simulation = QtWidgets.QPushButton("Load Simulation Data")
        self.btn_load_simulation.setStyleSheet("background-color: #E6CCFF; padding: 10px; font-weight: bold;")

        # Button for real-time mode
        self.btn_live_capture = QtWidgets.QPushButton("Live Capture (PSRT)")
        self.btn_live_capture.setStyleSheet("background-color: #FFA07A; padding: 10px; font-weight: bold;")

        l_source.addWidget(self.btn_load_offline)
        l_source.addWidget(self.btn_load_simulation)
        l_source.addWidget(self.btn_live_capture)
        
        l_source.addWidget(self.btn_load_offline)
        l_source.addWidget(self.btn_live_capture)
        grp_source.setLayout(l_source)
        
        panel_layout.addWidget(grp_source)
        panel_layout.addSpacing(15)
        # -------------------------

        self.btn_toggle_frames = QtWidgets.QPushButton("Close Frames")
        self.btn_show_y = QtWidgets.QPushButton("Show Y Heatmap")
        self.btn_show_beta_gamma = QtWidgets.QPushButton("Show β/γ Rotation")

        btn_font = QtGui.QFont("Microsoft JhengHei UI", 12, QtGui.QFont.Bold)
        self.btn_toggle_frames.setFont(btn_font)
        self.btn_show_y.setFont(btn_font)
        self.btn_show_beta_gamma.setFont(btn_font)

        panel_layout.addWidget(self.btn_toggle_frames)
        panel_layout.addWidget(self.btn_show_y)
        panel_layout.addWidget(self.btn_show_beta_gamma)

        
        # overlay checkboxes (bottom)
        self.chk_red_box = QtWidgets.QCheckBox("Show Red Box")
        self.chk_yellow_box = QtWidgets.QCheckBox("Show Yellow Band")
        self.chk_orange_grid = QtWidgets.QCheckBox("Show Orange 3x3 Grid")

        self.chk_red_box.setChecked(True)
        self.chk_yellow_box.setChecked(True)
        self.chk_orange_grid.setChecked(True)

        panel_layout.addSpacing(10)
        panel_layout.addWidget(self.chk_red_box)
        panel_layout.addWidget(self.chk_yellow_box)
        panel_layout.addWidget(self.chk_orange_grid)

        # --- NEW: Manual Labeling Control Group ---
        grp_label = QtWidgets.QGroupBox("Manual Labeling")
        l_label = QtWidgets.QVBoxLayout()

        # 1. Frame Selection Slider
        self.lbl_frame_idx = QtWidgets.QLabel("Frame: -")
        self.slider_frame = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setEnabled(False)  # Will be enabled after processing
        self.slider_frame.setSingleStep(10)
        self.slider_frame.setPageStep(10)
        self.slider_frame.setTickInterval(10)
        self.slider_frame.setTickPosition(QtWidgets.QSlider.TicksBelow)

        # smoother: only emit frame switch when mouse released
        self.slider_frame.setTracking(False)

        
        # 2. Labeling Actions
        self.btn_start_label = QtWidgets.QPushButton("Start Labeling")
        self.btn_start_label.setCheckable(True)
        self.btn_start_label.setStyleSheet("QPushButton:checked { background-color: #90EE90; }") 
        
        self.btn_clear_label = QtWidgets.QPushButton("Clear This Label")
        
        # 3. Generation Action
        self.btn_generate_3d = QtWidgets.QPushButton("Generate 3D Surface")
        self.btn_generate_3d.setStyleSheet("background-color: #FFB6C1; font-weight: bold; min-height: 30px;")

        l_label.addWidget(self.lbl_frame_idx)
        l_label.addWidget(self.slider_frame)
        l_label.addSpacing(10)
        l_label.addWidget(self.btn_start_label)
        l_label.addWidget(self.btn_clear_label)
        l_label.addSpacing(15)
        l_label.addWidget(self.btn_generate_3d)
        
        grp_label.setLayout(l_label)
        panel_layout.addWidget(grp_label)
        # ------------------------------------------

        panel_layout.addStretch(1)
        layout.addWidget(panel, stretch=0)

        # Controller (PyVista + callbacks)
        self.ctrl = VisualizerController(session=self.sess, cfg=self.cfg)
        self.ctrl.plotter = self.vtk_widget
        self.ctrl.heatmap_overlay_label = self.heatmap_overlay

        # Basic plotter settings
        self.ctrl.plotter.set_background("white")
        self.ctrl.plotter.enable_parallel_projection()

        # Connect buttons
        self.btn_toggle_frames.clicked.connect(self._on_toggle_frames)
        self.btn_show_y.clicked.connect(self._on_show_y_heatmap)
        self.btn_show_beta_gamma.clicked.connect(self._on_show_beta_gamma)
        self.chk_red_box.toggled.connect(self._on_toggle_red_box)
        self.chk_yellow_box.toggled.connect(self._on_toggle_yellow_band)
        self.chk_orange_grid.toggled.connect(self._on_toggle_orange_grid)

        # DO NOT build initial scene at startup, wait for user input
        # self.ctrl.build_initial_scene()

        # Connect data source buttons
        self.btn_load_offline.clicked.connect(self._on_load_offline)
        self.btn_load_simulation.clicked.connect(self._on_load_simulation)
        self.btn_live_capture.clicked.connect(self._on_live_capture)

        # --- NEW: Labeling Signals ---
        self.slider_frame.valueChanged.connect(self._on_slider_changed)
        self.slider_frame.sliderReleased.connect(self._on_slider_released)
        self.slider_frame.actionTriggered.connect(self._on_slider_action)
        self.btn_start_label.clicked.connect(self._on_label_mode_toggled)
        self.btn_clear_label.clicked.connect(self._on_clear_label)
        self.btn_generate_3d.clicked.connect(self._on_generate_3d)

        # --- NEW: debounce slider -> avoid heavy refresh spam ---
        self._label_slider_timer = QtCore.QTimer(self)
        self._label_slider_timer.setSingleShot(True)
        self._label_slider_timer.timeout.connect(self._apply_label_slider_value)

        # Timer to enable slider when data is ready
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._check_processing_done)
        self.timer.start(500)

    def _on_load_offline(self):
        """
        Triggered when 'Load Offline Videos' is clicked.
        Opens file dialogs to select local .avi/.mp4 files.
        """
        path_L, _ = QFileDialog.getOpenFileName(
            self, "Select Left Plane Video (Plane A)", "", "Video Files (*.avi *.mp4)"
        )
        if not path_L:
            return

        path_R, _ = QFileDialog.getOpenFileName(
            self, "Select Right Plane Video (Plane B)", "", "Video Files (*.avi *.mp4)"
        )
        if not path_R:
            return

        self.cfg.apply_mode_settings("video")
        self._load_videos_and_start(path_L, path_R)

    def _on_load_simulation(self):
        """
        Triggered when 'Load Simulation PNG Folders' is clicked.
        User selects LEFT folder first, then RIGHT folder.
        """
        left_dir = QFileDialog.getExistingDirectory(self, "Select LEFT PNG Folder")
        if not left_dir:
            return

        right_dir = QFileDialog.getExistingDirectory(self, "Select RIGHT PNG Folder")
        if not right_dir:
            return

        self.cfg.apply_mode_settings("simulation")
        self._load_simulation_and_start(left_dir, right_dir)

    def _on_live_capture(self):
        """
        Triggered when 'Live Capture' is clicked.
        Disables buttons and starts the PSRT script in a background thread.
        """
        # Disable buttons to prevent multiple clicks
        self.btn_live_capture.setEnabled(False)
        self.btn_load_offline.setEnabled(False)
        self.btn_load_simulation.setEnabled(False)
        self.btn_live_capture.setText("Scanning... Please Wait")
        self.ctrl.update_status("Connecting to Prodigy and capturing frames...")

        # Start the background thread for PSRT
        self.capture_thread = CaptureThread()
        self.capture_thread.finished_signal.connect(self._on_capture_finished)
        self.capture_thread.start()

    def _on_capture_finished(self, success: bool, message: str):
        """
        Callback triggered when the PSRT background thread finishes.
        """
        # Re-enable buttons
        self.btn_live_capture.setEnabled(True)
        self.btn_load_offline.setEnabled(True)
        self.btn_load_simulation.setEnabled(True)
        self.btn_live_capture.setText("Live Capture (PSRT)")

        if success:
            QMessageBox.information(self, "Success", "Live capture finished! Loading data...")
            self.cfg.apply_mode_settings("live")
            self._load_videos_and_start("L_s.avi", "R_s.avi")
        else:
            QMessageBox.critical(self, "Capture Failed", f"Error communicating with Prodigy:\n{message}")
            self.ctrl.update_status("Capture failed. Check connection.")

    def _load_videos_and_start(self, left_path: str, right_path: str):
        """
        Core function to extract frames from video files and initialize the 3D scene.
        """
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            QMessageBox.warning(self, "Error", "Video files not found!")
            return

        self.ctrl.update_status("Extracting frames from videos... please wait.")
        QtWidgets.QApplication.processEvents()

        try:
            loader = VideoLoader(output_fps=self.cfg.output_fps)

            self.sess.input_mode = self.cfg.input_mode
            self.sess.left_video_path = left_path
            self.sess.right_video_path = right_path
            self.sess.left_source_path = left_path
            self.sess.right_source_path = right_path

            self.sess.left_frames_original = loader.extract_frames(left_path)
            self.sess.right_frames_original = loader.extract_frames(right_path)

            if len(self.sess.left_frames_original) != len(self.sess.right_frames_original):
                raise ValueError(
                    f"Left/right frame count mismatch: "
                    f"{len(self.sess.left_frames_original)} vs {len(self.sess.right_frames_original)}"
                )

            self.sess.ensure_original_dims()
            self.ctrl.build_initial_scene()

            self.ctrl.update_status(
                f"Loaded {len(self.sess.right_frames_original)} frames "
                f"from {self.cfg.input_mode} mode."
            )

        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))
            self.ctrl.update_status("Failed to load video data.")

    def _load_simulation_and_start(self, left_dir: str, right_dir: str):
        """
        Core function to load LEFT/RIGHT PNG folders and initialize the 3D scene.
        """
        if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
            QMessageBox.warning(self, "Error", "Simulation folders not found!")
            return

        self.ctrl.update_status("Loading simulation PNG folders... please wait.")
        QtWidgets.QApplication.processEvents()

        try:
            loader = ImageSequenceLoader()

            self.sess.input_mode = self.cfg.input_mode
            self.sess.left_video_path = ""
            self.sess.right_video_path = ""
            self.sess.left_source_path = left_dir
            self.sess.right_source_path = right_dir

            self.sess.left_frames_original = loader.extract_frames_from_folder(left_dir)
            self.sess.right_frames_original = loader.extract_frames_from_folder(right_dir)

            nL = len(self.sess.left_frames_original)
            nR = len(self.sess.right_frames_original)

            if nL != nR:
                raise ValueError(f"Left/right PNG count mismatch: {nL} vs {nR}")

            if nL == 0:
                raise ValueError("No simulation PNG frames were loaded.")

            self.sess.ensure_original_dims()
            self.ctrl.build_initial_scene()

            self.ctrl.update_status(
                f"Loaded {nR} simulation frames. "
                f"crop_size={self.cfg.crop_size}, fh_dy_mm_per_frame={self.cfg.fh_dy_mm_per_frame}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Simulation Load Failed", str(e))
            self.ctrl.update_status("Failed to load simulation PNG folders.")

    def _on_toggle_frames(self):
        self.ctrl.toggle_frames()
        self.btn_toggle_frames.setText("Open Frames" if not self.sess.frames_visible else "Close Frames")

    def _on_show_y_heatmap(self):
        if not self.sess.selection_confirmed:
            QtWidgets.QMessageBox.information(
                self,
                "Show Y Heatmap",
                "Please finish ROI/crop selection and run processing first."
            )
            return
        self.ctrl.on_show_y_heatmap()

    def _on_toggle_red_box(self, checked: bool):
        self.ctrl.toggle_crop_box(checked)

    def _on_toggle_yellow_band(self, checked: bool):
        self.ctrl.toggle_band_box(checked)

    def _on_toggle_orange_grid(self, checked: bool):
        self.ctrl.toggle_grid9_box(checked)


    def _on_show_beta_gamma(self):
        if not self.sess.selection_confirmed:
            QtWidgets.QMessageBox.information(
                self,
                "Show β/γ Rotation",
                "Please finish ROI/crop selection and run processing first."
            )
            return
        self.ctrl.on_show_beta_gamma()

    # --- NEW METHODS FOR MANUAL LABELING ---

    def _check_processing_done(self):
        """Periodically checks if processing is done to enable the frame slider."""
        if self.sess.selection_confirmed and self.sess.right_frames is not None:
            n = len(self.sess.right_frames)
            if n > 0 and self.slider_frame.maximum() != n - 1:
                self.slider_frame.setMaximum(n - 1)
                self.slider_frame.setEnabled(True)
                self.slider_frame.setValue(0)
                self.lbl_frame_idx.setText(f"Frame: 0 / {n-1}")
                self.timer.stop()

    def _on_slider_changed(self, val):
        self.lbl_frame_idx.setText(f"Frame: {val} / {self.slider_frame.maximum()}")

    def _on_slider_released(self):
        if self.btn_start_label.isChecked():
            self._label_slider_timer.start(30)  # 30ms debounce


    def _apply_label_slider_value(self):
        if not self.btn_start_label.isChecked():
            return
        val = self.slider_frame.value()
        self.ctrl.set_active_labeling_frame(val)

    def _on_slider_action(self, action):
        if self.btn_start_label.isChecked():
            self._label_slider_timer.start(30)

    def _on_label_mode_toggled(self, checked):
        idx = self.slider_frame.value()
        if checked:
            self.slider_frame.setEnabled(True)

            # cursor like a pen/crosshair
            self.vtk_widget.setCursor(QtCore.Qt.CrossCursor)

            self.ctrl.start_labeling_mode(idx)
            self.btn_start_label.setText("Stop Labeling")
        else:
            self.ctrl.stop_labeling_mode()

            # restore cursor
            self.vtk_widget.setCursor(QtCore.Qt.ArrowCursor)

            self.btn_start_label.setText("Start Labeling")


    def _on_clear_label(self):
        idx = self.slider_frame.value()
        self.ctrl.clear_label_for_frame(idx)

    def _on_generate_3d(self):
        self.ctrl.generate_surface_from_labels()
