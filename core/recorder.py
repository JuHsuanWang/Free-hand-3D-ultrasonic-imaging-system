import cv2
import numpy as np
import os

class DualPlaneRecorder:
    def __init__(self, left_filename="L_s.avi", right_filename="R_s.avi", fps=10, frame_size=(512, 512)):
        """
        Initialize the dual-plane video recorder.
        """
        self.left_filename = left_filename
        self.right_filename = right_filename
        self.frame_size = frame_size
        
        # Use XVID codec for .avi format
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # isColor=False indicates we are writing single-channel grayscale images
        self.writer_L = cv2.VideoWriter(self.left_filename, fourcc, fps, self.frame_size, isColor=False)
        self.writer_R = cv2.VideoWriter(self.right_filename, fourcc, fps, self.frame_size, isColor=False)
        
        print(f"[Recorder] Initialized. Saving to {left_filename} and {right_filename}")

    def write_frames(self, img_L: np.ndarray, img_R: np.ndarray):
        """
        Write a single frame of dual-plane images.
        """
        # Ensure data format is correct (uint8)
        if img_L.dtype != np.uint8:
            img_L = img_L.astype(np.uint8)
        if img_R.dtype != np.uint8:
            img_R = img_R.astype(np.uint8)
            
        # Write frames to respective video files
        self.writer_L.write(img_L)
        self.writer_R.write(img_R)

    def release(self):
        """
        Release resources and finalize the video files.
        """
        self.writer_L.release()
        self.writer_R.release()
        print(f"[Recorder] Finished recording. Videos saved as {self.left_filename} and {self.right_filename}")