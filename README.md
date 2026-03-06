# Video Crop & 3D Visualizer

A PyQt5 + PyVista desktop tool for loading synchronized left/right ultrasound videos, selecting an ROI, choosing a crop center, stabilizing the sequence, visualizing stacked 3D frames, inspecting out-of-plane motion, and manually labeling contours to reconstruct a 3D surface and estimate volume.

This README is written for two purposes:

1. **Upload to GitHub** as the project homepage.
2. **Recreate the environment on another computer** (for example, your laptop).

---

## 1. Main features

- Load **left/right AVI videos** and sample frames at a target FPS.
- Select the ultrasound ROI interactively.
- Select a crop center and generate a **100×100 crop** plus a **3×3 grid** overlay.
- Optional **NCC + Kabsch stabilization** on the cropped region.
- Compute and display:
  - **Y heatmap** (out-of-plane displacement-like analysis)
  - **β/γ rotation** curves
- Build a stacked **3D PyVista visualization**.
- Support **manual labeling** on selected frames and reconstruct a **3D surface**.
- Estimate **surface volume** in mm³ / mL.
- Save cropped frames, stabilization debug images, and analysis outputs to an output folder.

---

## 2. Expected project structure

Recommended repository structure:

```text
your-repo/
├─ main.py
├─ config.py
├─ requirements.txt
├─ README.md
├─ core/
│  ├─ loader.py
│  └─ session.py
├─ gui/
│  ├─ window.py
│  └─ visualizer.py
├─ algorithms/
│  ├─ geometry.py
│  ├─ stabilizer.py
│  └─ out_of_plane.py
├─ analysis/
│  └─ gt_plot.py                 # optional but required if GT plot is enabled
├─ data/                         # optional
│  ├─ Ln.avi
│  ├─ Rn.avi
│  └─ tracker_data_1.csv
└─ output/
```

> Note: the current code imports modules using package paths such as `core.loader`, `gui.window`, and `algorithms.stabilizer`, so your GitHub repository should keep those folders instead of putting everything at the root.

---

## 3. System requirements

### Recommended

- **OS:** Windows 10 / Windows 11
- **Python:** 3.10
- **GPU:** optional

---

## 4. Create a fresh environment on a new computer

Use Conda as the standard setup flow:

```bash
git https://github.com/JuHsuanWang/Free-hand-3D-ultrasonic-imaging-system.git
cd Free-hand-3D-ultrasonic-imaging-system
conda create -n us3d python=3.10 -y
conda activate us3d
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Input files

At minimum, the program needs two video files:

- left video
- right video

By default, `main.py` looks for:

- `Ln.avi`
- `Rn.avi`

in the current working directory.

If you pass command-line arguments, those paths will be used instead.

### Optional input

- `tracker_data_1.csv`
  - used for GT comparison when `enable_gt_plot = True`
- `analysis/gt_plot.py`
  - required by the GT plotting code path

If you do not need EM/GT comparison, you can disable it in `config.py` by setting:

```python
enable_gt_plot = False
```

---

## 6. How to run

### Run with default filenames

Put `Ln.avi` and `Rn.avi` next to `main.py`, then run:

```bash
python main.py
```

### Run with explicit paths

```bash
python main.py path/to/left_video.avi path/to/right_video.avi
```

---

## 7. Workflow in the GUI

After launching:

1. **ROI selection**
   - Click the **top-left** corner of the ultrasound region.
   - Click the **bottom-right** corner.
   - Press **Enter** to confirm.

2. **Crop-center selection**
   - Click the crop center.
   - The program shows:
     - red center point
     - red central box
     - orange 3×3 grid
     - yellow middle band
   - Press **Enter** to start processing.

3. **Processing stage**
   - Optional stabilization
   - crop refresh
   - segmentation
   - output saving
   - optional GT plotting

4. **3D stage**
   - inspect stacked frames
   - show/hide overlays
   - show Y heatmap
   - show β/γ rotation

5. **Manual labeling / surface reconstruction**
   - Use the manual labeling tools to annotate the target contour on selected frames.
   - Generate a closed 3D surface from the labeled contours.
   - Compute the estimated volume.

---

## 8. Output folders

The program creates an output directory for each run:

```text
output/<run_name>/
```

Typical saved content may include:

- cropped left/right frames
- stabilization debug images
- segmentation or labeling outputs
- CSV / plots for motion analysis
- reconstructed surface results

The exact content depends on which options are enabled in `config.py` and which buttons/tools you use in the GUI.

---

## 9. Quick start

```bash
conda create -n us3d python=3.10 -y
conda activate us3d
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

