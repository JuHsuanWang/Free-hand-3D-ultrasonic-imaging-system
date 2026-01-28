# gui/window.py
from __future__ import annotations
from PyQt5 import QtWidgets, QtGui, QtCore
from pyvistaqt import QtInteractor

from core.session import SessionState
from config import AppConfig
from gui.visualizer import VisualizerController


class MainWindow(QtWidgets.QMainWindow):
    """PyQt main window: embeds PyVista interactor and provides control buttons."""

    def __init__(self, session: SessionState, cfg: AppConfig):
        super().__init__()
        self.sess = session
        self.cfg = cfg

        self.setWindowTitle("Video Crop & 3D Visualizer (Modular)")
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

        self.btn_toggle_frames = QtWidgets.QPushButton("Close Frames")
        self.btn_toggle_pc = QtWidgets.QPushButton("Close Point Cloud")
        self.btn_show_y = QtWidgets.QPushButton("Show Y Heatmap")
        self.btn_show_beta_gamma = QtWidgets.QPushButton("Show β/γ Rotation")

        btn_font = QtGui.QFont("Microsoft JhengHei UI", 12, QtGui.QFont.Bold)
        self.btn_toggle_frames.setFont(btn_font)
        self.btn_toggle_pc.setFont(btn_font)
        self.btn_show_y.setFont(btn_font)
        self.btn_show_beta_gamma.setFont(btn_font)

        panel_layout.addWidget(self.btn_toggle_frames)
        panel_layout.addWidget(self.btn_toggle_pc)
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
        self.btn_toggle_pc.clicked.connect(self._on_toggle_point_cloud)
        self.btn_show_y.clicked.connect(self._on_show_y_heatmap)
        self.btn_show_beta_gamma.clicked.connect(self._on_show_beta_gamma)
        self.chk_red_box.toggled.connect(self._on_toggle_red_box)
        self.chk_yellow_box.toggled.connect(self._on_toggle_yellow_band)
        self.chk_orange_grid.toggled.connect(self._on_toggle_orange_grid)

        # Build initial scene
        self.ctrl.build_initial_scene()

    def _on_toggle_frames(self):
        self.ctrl.toggle_frames()
        self.btn_toggle_frames.setText("Open Frames" if not self.sess.frames_visible else "Close Frames")

    def _on_toggle_point_cloud(self):
        self.ctrl.toggle_point_cloud()
        self.btn_toggle_pc.setText("Open Point Cloud" if not self.sess.point_cloud_visible else "Close Point Cloud")

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
