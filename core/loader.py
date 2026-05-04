# core/loader.py
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class VideoLoader:
    """Video I/O layer: decode video and sample frames at target FPS."""

    def __init__(self, output_fps: int = 30):
        self.output_fps = output_fps
        self.decord_available = False
        self.decord_ctx = None

        # Try to import decord (optional).
        try:
            from decord import cpu, gpu  # noqa: F401
            self.decord_available = True
            self._init_decord_context()
            print("Decord available")
        except Exception:
            self.decord_available = False
            print("Decord not found - using OpenCV")

    def _init_decord_context(self):
        """Initialize decord context (GPU first, fallback to CPU)."""
        from decord import cpu, gpu

        try:
            self.decord_ctx = gpu(0)
            print("  Decord GPU mode")
        except Exception:
            self.decord_ctx = cpu(0)
            print("  Decord CPU mode")

    def extract_frames(self, video_path: str) -> np.ndarray:
        """Extract sampled frames from a video into (N,H,W,3) uint8 BGR."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        if self.decord_available:
            try:
                return self._extract_frames_decord(video_path)
            except Exception as e:
                print(f"  Decord failed: {e}")
                return self._extract_frames_opencv(video_path)

        return self._extract_frames_opencv(video_path)

    def _extract_frames_decord(self, video_path: str) -> np.ndarray:
        from decord import VideoReader, cpu

        try:
            vr = VideoReader(video_path, ctx=self.decord_ctx)
        except Exception:
            self.decord_ctx = cpu(0)
            vr = VideoReader(video_path, ctx=self.decord_ctx)

        total_frames = len(vr)
        fps = vr.get_avg_fps() or 30.0
        duration = total_frames / fps
        target_count = max(1, int(duration * self.output_fps))
        indices = np.linspace(0, total_frames - 1, target_count, dtype=int)

        print(f"Loading {video_path}... ({target_count} frames)")
        frames = vr.get_batch(indices).asnumpy()
        # decord returns RGB; convert to BGR to match OpenCV style if needed
        return frames[:, :, :, ::-1].copy()

    def _extract_frames_opencv(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps
        target_count = max(1, int(duration * self.output_fps))

        # IMPORTANT: use sorted array, not set(), for deterministic order + seeking
        indices = np.linspace(0, max(0, total_frames - 1), target_count, dtype=int)
        indices = np.unique(indices)  # ensure strictly increasing and unique

        print(f"Loading {video_path}... ({len(indices)} frames)")

        frames = (
            self._read_dense_sample(cap, indices)
            if self._should_read_dense(total_frames, len(indices))
            else self._read_sparse_sample(cap, indices)
        )

        cap.release()
        return np.asarray(frames, dtype=np.uint8)

    @staticmethod
    def _should_read_dense(total_frames: int, sample_count: int) -> bool:
        if total_frames <= 0:
            return False
        return (float(sample_count) / float(total_frames)) >= 0.08

    @staticmethod
    def _read_sparse_sample(cap: cv2.VideoCapture, indices: np.ndarray) -> list:
        frames = []
        last_pos = -1
        for idx in indices:
            if idx != last_pos + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

            ret, frame = cap.read()
            if not ret:
                ok = False
                for _ in range(3):
                    ret2, frame2 = cap.read()
                    if ret2:
                        frame = frame2
                        ok = True
                        break
                if not ok:
                    break

            frames.append(frame)
            last_pos = int(idx)
        return frames

    @staticmethod
    def _read_dense_sample(cap: cv2.VideoCapture, indices: np.ndarray) -> list:
        frames = []
        targets = set(int(v) for v in indices)
        max_idx = int(indices[-1]) if len(indices) else -1
        frame_idx = 0
        while frame_idx <= max_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in targets:
                frames.append(frame)
            frame_idx += 1
        return frames

class ImageSequenceLoader:
    """Load ordered PNG/JPG image sequences from folders into (N,H,W,3) uint8 BGR."""

    def __init__(self, valid_exts=None):
        if valid_exts is None:
            valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.valid_exts = tuple(ext.lower() for ext in valid_exts)

    def extract_frames_from_folder(self, folder_path: str) -> np.ndarray:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        filenames = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(self.valid_exts)
        ]
        filenames = sorted(filenames)

        if len(filenames) == 0:
            raise ValueError(f"No image files found in folder: {folder_path}")

        paths = [os.path.join(folder_path, fname) for fname in filenames]

        def _read_one(fpath: str) -> np.ndarray:
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {fpath}")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.ndim == 3 and img.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unsupported image shape {img.shape} for file: {fpath}")
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return img

        frames = []
        ref_h, ref_w = None, None

        print(f"Loading image sequence from {folder_path}... ({len(filenames)} frames)")

        workers = min(8, max(1, os.cpu_count() or 1), len(paths))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            loaded = list(ex.map(_read_one, paths))

        for fname, img in zip(filenames, loaded):
            h, w = img.shape[:2]
            if ref_h is None:
                ref_h, ref_w = h, w
            elif h != ref_h or w != ref_w:
                raise ValueError(
                    f"Image size mismatch in folder {folder_path}. "
                    f"Expected {(ref_h, ref_w)}, got {(h, w)} for {fname}"
                )

            frames.append(img)

        return np.asarray(frames, dtype=np.uint8)
