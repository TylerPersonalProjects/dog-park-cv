"""
Core detection engine for dog park CV system.
Handles person + dog detection, squat pose analysis, and clip triggering.
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO

PERSON_CLASS = 0
DOG_CLASS = 16


@dataclass
class TrackedEntity:
    track_id: int
    cls: int
    bbox: tuple
    center: tuple
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    squat_start: Optional[float] = None
    squat_confirmed: bool = False
    last_seen: float = field(default_factory=time.time)

    def update(self, bbox, pose_kpts=None):
        self.bbox = bbox
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.center = (cx, cy)
        self.history.append(self.center)
        self.last_seen = time.time()
        if pose_kpts is not None:
            self._check_squat(bbox, pose_kpts)

    def _check_squat(self, bbox, kpts):
        if kpts is None or len(kpts) < 4:
            return
        x1, y1, x2, y2 = bbox
        box_h = max(y2 - y1, 1)
        rear_y_ratio = (y2 - y1 * 0.25) / box_h
        aspect = (y2 - y1) / max(x2 - x1, 1)
        is_squatting = aspect > 0.7 and rear_y_ratio > 0.85
        if is_squatting:
            if self.squat_start is None:
                self.squat_start = time.time()
            elif time.time() - self.squat_start >= 1.5:
                self.squat_confirmed = True
        else:
            self.squat_start = None
            self.squat_confirmed = False

    def distance_to(self, other) -> float:
        return np.hypot(self.center[0] - other.center[0],
                        self.center[1] - other.center[1])

    def is_bending(self) -> bool:
        if len(self.history) < 10:
            return False
        recent = list(self.history)[-10:]
        dy = recent[-1][1] - recent[0][1]
        return dy > 15


@dataclass
class IncidentEvent:
    dog_id: int
    person_id: Optional[int]
    timestamp: float
    frame_count: int
    clip_saved: bool = False
    clip_path: str = ""


class DogParkDetector:
    PICKUP_WINDOW = 20.0
    PRE_BUFFER_SECS = 5
    CLIP_DURATION_SECS = 15
    PROXIMITY_PX = 200
    STALE_TRACK_SECS = 3.0

    def __init__(self, model_path: str = "yolov8n.pt",
                 pose_model_path: str = "yolov8n-pose.pt"):
        print("[Detector] Loading YOLO models...")
        self.model = YOLO(model_path)
        self.pose_model = YOLO(pose_model_path)
        self.tracks = {}
        self.incidents = []
        self.pending_incidents = {}
        self._lock = threading.Lock()
        self.frame_count = 0
        self.on_incident_callback = None
        print("[Detector] Models loaded.")

    def process_frame(self, frame: np.ndarray):
        self.frame_count += 1
        annotated = frame.copy()
        new_incidents = []

        det_results = self.model.track(
            frame, persist=True, verbose=False,
            classes=[PERSON_CLASS, DOG_CLASS], conf=0.4, iou=0.5
        )
        pose_results = self.pose_model(frame, verbose=False, conf=0.35)

        with self._lock:
            self._update_tracks(det_results, pose_results)
            self._check_incidents(new_incidents)
            self._draw_annotations(annotated)
            self._cleanup_stale()

        return annotated, new_incidents

    def _update_tracks(self, det_results, pose_results):
        if not det_results or det_results[0].boxes is None:
            return
        boxes = det_results[0].boxes
        if boxes.id is None:
            return

        pose_kpts_list = []
        if pose_results and pose_results[0].keypoints is not None:
            pose_kpts_list = pose_results[0].keypoints.xy.cpu().numpy()

        for box, track_id, cls_id, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy()
        ):
            if conf < 0.4:
                continue
            bbox = tuple(map(int, box))
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2

            pose_kpts = None
            if cls_id == DOG_CLASS and pose_kpts_list:
                for kpts in pose_kpts_list:
                    if len(kpts) > 0:
                        kx = np.mean(kpts[:, 0][kpts[:, 0] > 0])
                        ky = np.mean(kpts[:, 1][kpts[:, 1] > 0])
                        if abs(kx - cx) < 80 and abs(ky - cy) < 80:
                            pose_kpts = kpts
                            break

            if track_id in self.tracks:
                self.tracks[track_id].update(bbox, pose_kpts if cls_id == DOG_CLASS else None)
            else:
                entity = TrackedEntity(track_id=track_id, cls=cls_id, bbox=bbox, center=(cx, cy))
                entity.update(bbox, pose_kpts if cls_id == DOG_CLASS else None)
                self.tracks[track_id] = entity

    def _check_incidents(self, new_incidents):
        dogs = {tid: t for tid, t in self.tracks.items() if t.cls == DOG_CLASS}
        people = {tid: t for tid, t in self.tracks.items() if t.cls == PERSON_CLASS}

        for dog_id, dog in dogs.items():
            if not dog.squat_confirmed:
                continue
            if dog_id in self.pending_incidents:
                incident = self.pending_incidents[dog_id]
                elapsed = time.time() - incident.timestamp
                nearby = [p for p in people.values() if p.distance_to(dog) < self.PROXIMITY_PX]
                picked_up = any(p.is_bending() for p in nearby)
                if picked_up:
                    del self.pending_incidents[dog_id]
                    continue
                if elapsed >= self.PICKUP_WINDOW:
                    new_incidents.append(incident)
                    self.incidents.append(incident)
                    del self.pending_incidents[dog_id]
                    if self.on_incident_callback:
                        self.on_incident_callback(incident)
            else:
                nearest_person_id = None
                min_dist = float("inf")
                for pid, person in people.items():
                    d = person.distance_to(dog)
                    if d < min_dist:
                        min_dist = d
                        nearest_person_id = pid
                self.pending_incidents[dog_id] = IncidentEvent(
                    dog_id=dog_id, person_id=nearest_person_id,
                    timestamp=time.time(), frame_count=self.frame_count
                )

    def _draw_annotations(self, frame):
        for entity in self.tracks.values():
            x1, y1, x2, y2 = entity.bbox
            if entity.cls == PERSON_CLASS:
                color = (0, 220, 80)
                label = f"Person #{entity.track_id}"
                thickness = 2
            else:
                color = (200, 60, 200)
                label = f"Dog #{entity.track_id}"
                thickness = 2
                if entity.squat_confirmed:
                    color = (0, 80, 255)
                    label = f"Dog #{entity.track_id} WARNING SQUATTING"
                    thickness = 3

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        for dog_id, incident in self.pending_incidents.items():
            if dog_id in self.tracks:
                dog = self.tracks[dog_id]
                elapsed = time.time() - incident.timestamp
                remaining = max(0, self.PICKUP_WINDOW - elapsed)
                x1, y1, x2, y2 = dog.bbox
                cv2.putText(frame, f"No pickup: {remaining:.0f}s", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Incidents today: {len(self.incidents)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def _cleanup_stale(self):
        now = time.time()
        stale = [tid for tid, t in self.tracks.items() if now - t.last_seen > self.STALE_TRACK_SECS]
        for tid in stale:
            del self.tracks[tid]

    def get_track_summary(self) -> dict:
        with self._lock:
            people = sum(1 for t in self.tracks.values() if t.cls == PERSON_CLASS)
            dogs = sum(1 for t in self.tracks.values() if t.cls == DOG_CLASS)
            squatting = sum(1 for t in self.tracks.values() if t.cls == DOG_CLASS and t.squat_confirmed)
            return {
                "people": people, "dogs": dogs, "squatting": squatting,
                "pending_incidents": len(self.pending_incidents),
                "total_incidents": len(self.incidents),
            }
