"""
Camera manager: wraps OpenCV VideoCapture for the USB Logitech camera.
Runs capture in a background thread so the main loop never blocks on reads.
"""

import cv2
import time
import threading
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, camera_index: int = 0, width: int = 1280,
                 height: int = 720, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._annotated_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.dropped_frames = 0
        self._actual_fps = fps

    def start(self) -> bool:
        logger.info(f"Opening camera {self.camera_index}...")
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            for idx in range(1, 5):
                self._cap = cv2.VideoCapture(idx)
                if self._cap.isOpened():
                    logger.info(f"Camera found at index {idx}")
                    break
            else:
                logger.error("No camera found. Check USB connection.")
                return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: {actual_w}x{actual_h}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        for _ in range(50):
            if self._raw_frame is not None:
                break
            time.sleep(0.1)
        return True

    def _capture_loop(self):
        t_last = time.time()
        fps_counter = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self.dropped_frames += 1
                time.sleep(0.05)
                continue
            with self._lock:
                self._raw_frame = frame
            self.frame_count += 1
            fps_counter += 1
            now = time.time()
            if now - t_last >= 5.0:
                self._actual_fps = fps_counter / (now - t_last)
                fps_counter = 0
                t_last = now

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._raw_frame.copy() if self._raw_frame is not None else None

    def set_annotated_frame(self, frame: np.ndarray):
        with self._lock:
            self._annotated_frame = frame

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._annotated_frame is not None:
                return self._annotated_frame.copy()
            return self._raw_frame.copy() if self._raw_frame is not None else None

    def get_resolution(self) -> tuple:
        if self._raw_frame is not None:
            h, w = self._raw_frame.shape[:2]
            return w, h
        return self.width, self.height

    def stop(self):
        logger.info("Stopping camera...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("Camera stopped.")

    @property
    def actual_fps(self) -> float:
        return self._actual_fps
