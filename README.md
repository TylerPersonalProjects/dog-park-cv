a# 🐾 Dog Park CV — Computer Vision Monitor

A 24/7 computer vision system that monitors a dog park, detects when dogs poop, and saves evidence clips when owners don't pick it up.

## What it does
- **Live detection** — person (green box) and dog (pink box) tracking with YOLOv8
- - **Squat detection** — pose estimation detects when a dog is pooping
  - - **Pickup logic** — 20 second window to see if owner cleans up
    - - **Clip saving** — auto-saves 15s clips when no pickup detected
      - - **Live dashboard** — browser view at http://localhost:5000
        - - **Google Drive upload** — optional automatic clip backup
         
          - ## Quick Start
          - ```bash
            pip install -r requirements.txt
            python main.py
            ```
            Open http://localhost:5000

            ## Options
            ```bash
            python main.py --camera 1       # different camera index
            python main.py --no-drive       # local storage only
            python main.py --model yolov8s.pt  # better accuracy
            ```

            ## Stack
            - YOLOv8 (ultralytics) — detection + pose
            - - OpenCV — capture + annotation
              - - Flask — live MJPEG dashboard
                - - Google Drive API — optional clip upload
