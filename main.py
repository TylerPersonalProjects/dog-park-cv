"""
Dog Park Computer Vision System - Main Entry Point

Usage:
    python main.py                     # defaults
    python main.py --camera 1         # use camera index 1
    python main.py --no-drive         # local storage only
    python main.py --port 8080        # different dashboard port
    python main.py --width 1920 --height 1080
"""

import argparse
import logging
import signal
import sys
import time
import threading

from src.camera import CameraManager
from src.detector import DogParkDetector
from src.clip_manager import ClipManager
from src.web_app import init_app, run_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("dogpark_cv.log")]
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Dog Park CV System")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--clips-dir", default="clips")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--no-drive", action="store_true")
    p.add_argument("--drive-folder", default="DogParkCV_Incidents")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--pose-model", default="yolov8n-pose.pt")
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("  Dog Park CV System - Starting up")
    logger.info("=" * 60)

    drive_uploader = None
    if not args.no_drive:
        try:
            from src.drive_uploader import DriveUploader
            drive_uploader = DriveUploader(folder_name=args.drive_folder)
            if not drive_uploader.is_ready():
                logger.warning("Drive not ready. Using local storage only.")
                drive_uploader = None
        except Exception as e:
            logger.warning(f"Drive init failed: {e}. Using local only.")

    camera = CameraManager(camera_index=args.camera, width=args.width,
                           height=args.height, fps=args.fps * 2)
    if not camera.start():
        logger.error("Failed to open camera. Exiting.")
        sys.exit(1)
    logger.info(f"Camera ready: {camera.get_resolution()}")

    detector = DogParkDetector(model_path=args.model, pose_model_path=args.pose_model)

    clip_manager = ClipManager(output_dir=args.clips_dir, fps=args.fps,
                               drive_uploader=drive_uploader,
                               pre_roll_secs=5, post_roll_secs=10)

    def on_incident(incident):
        logger.warning(f"INCIDENT: Dog #{incident.dog_id} - no pickup detected!")
        clip_manager.save_incident_clip(incident_ts=incident.timestamp,
                                        dog_id=incident.dog_id,
                                        person_id=incident.person_id)

    detector.on_incident_callback = on_incident
    init_app(camera, detector, clip_manager)

    server_thread = threading.Thread(target=run_server,
                                     kwargs={"host": args.host, "port": args.port},
                                     daemon=True)
    server_thread.start()
    logger.info(f"Dashboard at http://localhost:{args.port}")

    shutdown_event = threading.Event()
    def handle_signal(sig, frame):
        shutdown_event.set()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    target_interval = 1.0 / args.fps
    while not shutdown_event.is_set():
        t_start = time.time()
        frame = camera.get_raw_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        clip_manager.push_frame(frame)
        annotated, _ = detector.process_frame(frame)
        camera.set_annotated_frame(annotated)
        time.sleep(max(0, target_interval - (time.time() - t_start)))

    camera.stop()
    logger.info(f"Done. Incidents: {len(detector.incidents)}, Clips: {clip_manager.get_clip_count()}")


if __name__ == "__main__":
    main()
