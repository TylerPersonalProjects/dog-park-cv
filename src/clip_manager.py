"""
Clip manager: rolling video buffer + saves incident clips.
Supports local storage and optional Google Drive upload.
"""

import cv2
import os
import time
import threading
import queue
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RollingBuffer:
    def __init__(self, max_seconds: int = 20, fps: int = 15):
        self.fps = fps
        self.max_frames = max_seconds * fps
        self._buf: deque = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

    def push(self, frame):
        with self._lock:
            self._buf.append((time.time(), frame.copy()))

    def get_frames_since(self, since_ts: float, extra_secs: float = 10.0):
        with self._lock:
            cutoff = since_ts - extra_secs
            return [(ts, f) for ts, f in self._buf if ts >= cutoff]


class ClipManager:
    CODEC = "mp4v"
    EXT = ".mp4"

    def __init__(self, output_dir: str = "clips", fps: int = 15,
                 drive_uploader=None, pre_roll_secs: int = 5,
                 post_roll_secs: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.drive_uploader = drive_uploader
        self.pre_roll_secs = pre_roll_secs
        self.post_roll_secs = post_roll_secs
        self.buffer = RollingBuffer(
            max_seconds=pre_roll_secs + post_roll_secs + 5, fps=fps)
        self._save_queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._save_worker, daemon=True)
        self._worker.start()
        self._saved_clips = []
        self._lock = threading.Lock()
        logger.info(f"ClipManager ready. Output: {self.output_dir.resolve()}")

    def push_frame(self, frame):
        self.buffer.push(frame)

    def save_incident_clip(self, incident_ts: float, dog_id: int,
                           person_id: Optional[int] = None, **kwargs):
        self._save_queue.put({
            "incident_ts": incident_ts, "dog_id": dog_id,
            "person_id": person_id, "queued_at": time.time()
        })

    def _save_worker(self):
        while True:
            job = self._save_queue.get()
            try:
                path = self._write_clip(job)
                if path and self.drive_uploader:
                    threading.Thread(target=self.drive_uploader.upload,
                                     args=(path,), daemon=True).start()
            except Exception as e:
                logger.error(f"Clip save failed: {e}", exc_info=True)
            finally:
                self._save_queue.task_done()

    def _write_clip(self, job: dict) -> Optional[str]:
        incident_ts = job["incident_ts"]
        dog_id = job["dog_id"]
        time.sleep(self.post_roll_secs)
        frames_with_ts = self.buffer.get_frames_since(
            incident_ts, extra_secs=self.pre_roll_secs)

        if not frames_with_ts:
            logger.warning("No frames in buffer for clip.")
            return None

        ts_str = datetime.fromtimestamp(incident_ts).strftime("%Y%m%d_%H%M%S")
        filename = f"incident_{ts_str}_dog{dog_id}{self.EXT}"
        filepath = self.output_dir / filename

        h, w = frames_with_ts[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.CODEC)
        writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (w, h))
        for _, frame in frames_with_ts:
            writer.write(frame)
        writer.release()

        duration = len(frames_with_ts) / self.fps
        size_kb = filepath.stat().st_size // 1024
        logger.info(f"Clip saved: {filepath} ({duration:.1f}s, {size_kb}KB)")

        clip_info = {
            "path": str(filepath), "filename": filename,
            "timestamp": incident_ts,
            "datetime": ts_str, "dog_id": dog_id,
            "person_id": job.get("person_id"),
            "duration_secs": duration, "size_kb": size_kb
        }
        with self._lock:
            self._saved_clips.append(clip_info)
        return str(filepath)

    def get_saved_clips(self) -> list:
        with self._lock:
            return list(reversed(self._saved_clips))

    def get_clip_count(self) -> int:
        with self._lock:
            return len(self._saved_clips)
