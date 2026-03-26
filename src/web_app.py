"""
Flask dashboard for dog park CV system.
Serves live annotated video stream + incident log + stats.
"""

import cv2
import time
import logging
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, send_from_directory
from pathlib import Path

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../templates", static_folder="../static")

_camera = None
_detector = None
_clip_manager = None
_start_time = time.time()


def init_app(camera, detector, clip_manager):
    global _camera, _detector, _clip_manager
    _camera = camera
    _detector = detector
    _clip_manager = clip_manager


def _generate_frames():
    while True:
        if _camera is None:
            time.sleep(0.1)
            continue
        frame = _camera.get_annotated_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")
        time.sleep(1 / 20)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(_generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    uptime = int(time.time() - _start_time)
    hours, rem = divmod(uptime, 3600)
    mins, secs = divmod(rem, 60)
    stats = _detector.get_track_summary() if _detector else {}
    clip_count = _clip_manager.get_clip_count() if _clip_manager else 0
    return jsonify({
        "uptime": f"{hours:02d}:{mins:02d}:{secs:02d}",
        "people_in_frame": stats.get("people", 0),
        "dogs_in_frame": stats.get("dogs", 0),
        "dogs_squatting": stats.get("squatting", 0),
        "pending_incidents": stats.get("pending_incidents", 0),
        "total_incidents_today": stats.get("total_incidents", 0),
        "clips_saved": clip_count,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })


@app.route("/api/incidents")
def api_incidents():
    if not _clip_manager:
        return jsonify([])
    return jsonify(_clip_manager.get_saved_clips()[:50])


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    clips_dir = Path("clips").resolve()
    return send_from_directory(str(clips_dir), filename)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    logger.info(f"Dashboard running at http://localhost:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
