# Autonomous Driving Perception & Driver Assistance System

A real-time driver assistance system built in Python that processes dashcam video and provides live feedback through an on-screen HUD dashboard.

**Author:** Aarnav  
**Programme:** B.Tech. Artificial Intelligence & Data Science — 8th Semester  
**Institution:** University School of Automation and Robotics (USAR), GGSIPU, New Delhi  

---

## Features

- **Vehicle Detection** — Real-time detection of cars, trucks, buses, and motorcycles using YOLOv8
- **Forward Collision Warning (FCW)** — Two-level alert system (Warning / Critical) based on vehicle proximity
- **Speed Estimation** — Centroid-based tracker estimates vehicle speed in km/h
- **Lane Detection** — Classical computer vision pipeline with temporal smoothing
- **Traffic Sign Recognition** — YOLO-based detection with optional custom CNN classifier
- **Night Mode** — Automatic CLAHE enhancement in low-light conditions
- **HUD Dashboard** — Semi-transparent live overlay showing all system outputs
- **Video Saving** — Processed output saved to `outputs/processed.mp4`

---

## Project Structure

```
adas-project/
├── src/
│   ├── main.py                      # Main pipeline and HUD renderer
│   ├── lane_detection.py            # Lane detection module
│   ├── vehicle_detection.py         # Vehicle detection and tracking
│   ├── traffic_sign_recognition.py  # Traffic sign recognition
│   └── train_sign_model.py          # CNN training script
├── data/
│   └── videos/
│       └── traffic.mp4              # Input dashcam video
├── models/                          # Trained CNN models (if available)
├── outputs/                         # Processed output video
├── config.yaml                      # All system parameters
├── yolov8n.pt                       # YOLOv8 nano model weights
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

---

## Installation

```bash
# Clone or download the project
cd adas-project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run with default video (data/videos/traffic.mp4)
python3 src/main.py

# Run with a custom video
python3 src/main.py path/to/your/video.mp4
```

**Keyboard Controls:**
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `Space` | Pause / Resume |

---

## Configuration

All parameters are controlled through `config.yaml` — no source code changes needed.

| Parameter | Location | Description |
|-----------|----------|-------------|
| `departure_offset_threshold` | `lane_detection` | Pixels of offset before lane departure alert |
| `conf_threshold` | `vehicle_detection` | Minimum detection confidence (0–1) |
| `fcw_critical_ratio` | `vehicle_detection` | Bounding box ratio for critical FCW alert |
| `brightness_threshold` | `night_mode` | Mean brightness below which CLAHE activates |
| `smoothing_alpha` | `lane_detection` | Temporal smoothing factor for lane lines |

---

## System Pipeline

```
Video Input
    ↓
Night Mode Pre-processing (CLAHE if dark)
    ↓
Lane Detection (HLS mask → Canny → Hough)
    ↓
Vehicle Detection & Tracking (YOLOv8 + Centroid Tracker)
    ↓
Traffic Sign Recognition (YOLO + CNN)
    ↓
HUD Renderer
    ↓
Display + Save Output
```

---

## Results

- Processing speed: ~13–14 FPS on MacBook Air M2 (CPU only)
- Vehicle detection confidence threshold: 0.4
- Speed estimation range: 20–60 km/h on test footage
- Night mode: auto-activates when mean brightness < 60/255

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.9 | Core language |
| OpenCV 4.x | Image processing and computer vision |
| YOLOv8n (Ultralytics) | Real-time object detection |
| TensorFlow / Keras | Custom CNN training |
| NumPy | Numerical computing |
| PyYAML | Configuration management |

---

## Future Scope

- Replace Hough-based lane detection with LaneNet / SCNN
- Train custom CNN on Indian traffic sign dataset
- Add monocular depth estimation (MiDaS)
- Deploy on NVIDIA Jetson Nano for in-vehicle use
- Add pedestrian and cyclist detection
- Driver drowsiness monitoring via facial landmarks
