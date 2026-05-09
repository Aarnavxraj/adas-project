import cv2
import sys
import time
import yaml
import numpy as np
from lane_detection import detect_lanes
from vehicle_detection import VehicleDetector
from traffic_sign_recognition import TrafficSignRecognizer
from pedestrian_risk import PedestrianRiskDetector

with open("config.yaml", "r") as f:
    _config = yaml.safe_load(f)
    _vcfg = _config["video"]
    _hcfg = _config["hud"]
    _scfg = _config["sign_recognition"]
    _pcfg = _config["pedestrian_risk"]

def draw_hud(frame, lane_info, vehicle_info, sign_info, ped_info, fps, paused, last_sign_label, last_sign_timer):
    h, w = frame.shape[:2]
    hud_w = _hcfg["width"]
    hud_h = _hcfg["height"]
    x, y = w - hud_w - 10, 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + hud_w, y + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, _hcfg["background_alpha"], frame, 1 - _hcfg["background_alpha"], 0, frame)

    def put(text, row, color=(255, 255, 255)):
        cv2.putText(frame, text, (x + 10, y + 25 + row * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS
    put(f"FPS: {fps:.1f}", 0)

    # Lane offset
    if lane_info["offset"] is not None:
        put(f"Offset: {lane_info['offset']}px", 1)
    else:
        put("Offset: --", 1, (100, 100, 100))

    # Lane status
    if lane_info["departure"]:
        put("LANE DEPARTURE", 2, (0, 0, 255))
    elif lane_info["lines_found"]:
        put("Lane: OK", 2, (0, 255, 0))
    else:
        put("Lane: Searching...", 2, (0, 200, 255))

    # FCW
    fcw = vehicle_info["fcw_level"]
    if fcw == "critical":
        put("FCW: CRITICAL", 3, (0, 0, 255))
    elif fcw == "warning":
        put("FCW: WARNING", 3, (0, 165, 255))
    else:
        put("FCW: Clear", 3, (0, 255, 0))

    # Nearest vehicle speed
    max_speed = vehicle_info.get("max_speed_kmh", 0)
    if max_speed > 0:
        put(f"Nearest: {max_speed:.0f} km/h", 4, (255, 255, 255))
    else:
        put("Nearest: -- km/h", 4, (100, 100, 100))

    # Last detected sign (persists for sign_display_frames frames)
    if last_sign_label and last_sign_timer > 0:
        put(f"Sign: {last_sign_label}", 5, (255, 100, 0))
    else:
        put("Sign: None", 5, (100, 100, 100))

    # Pedestrian risk
    ped_risk = ped_info.get("risk_level", "none")
    n_peds   = len(ped_info.get("pedestrians", []))
    if ped_risk == "danger":
        put(f"Pedestrian: DANGER ({n_peds})", 6, (0, 0, 255))
    elif ped_risk == "caution":
        put(f"Pedestrian: Caution ({n_peds})", 6, (0, 165, 255))
    elif n_peds > 0:
        put(f"Pedestrian: OK ({n_peds})", 6, (0, 255, 200))
    else:
        put("Pedestrian: None", 6, (100, 100, 100))

    # Night mode indicator
    if lane_info.get("night_mode"):
        put("Night Mode: ON", 7, (255, 200, 0))

    # Paused indicator
    if paused:
        put("[ PAUSED ]", 8, (0, 200, 255))

    # Border
    cv2.rectangle(frame, (x, y), (x + hud_w, y + hud_h), (80, 80, 80), 1)

def draw_warnings(frame, lane_info, vehicle_info, ped_info):
    h, w = frame.shape[:2]

    if lane_info["departure"]:
        cv2.putText(frame, "!! LANE DEPARTURE !!", (w // 2 - 200, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    if vehicle_info["fcw_level"] == "critical":
        cv2.putText(frame, "!! COLLISION WARNING !!", (w // 2 - 220, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif vehicle_info["fcw_level"] == "warning":
        cv2.putText(frame, "Vehicle Ahead", (w // 2 - 120, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

    if ped_info.get("risk_level") == "danger":
        cv2.putText(frame, "!! PEDESTRIAN DANGER !!", (w // 2 - 220, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif ped_info.get("risk_level") == "caution":
        cv2.putText(frame, "Pedestrian Ahead", (w // 2 - 140, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else _vcfg["default_path"]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if _vcfg["save_output"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(_vcfg["output_path"], fourcc, fps, (width, height))
        print(f"Saving output to: {_vcfg['output_path']}")

    detector    = VehicleDetector(fps=fps)
    recognizer  = TrafficSignRecognizer() if _scfg["enabled"] else None
    ped_detector = PedestrianRiskDetector()
    print("Video opened successfully. Press q to quit, space to pause/resume.")

    prev_time = time.time()
    paused = False
    last_sign_label = None
    last_sign_timer = 0

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

        if paused:
            cv2.waitKey(100)
            continue

        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame.")
            break

        lane_output, lane_info = detect_lanes(frame)
        vehicle_output, vehicle_info = detector.detect_vehicles(lane_output)

        sign_info = {"signs": [], "last_sign": None}
        if recognizer:
            sign_output, sign_info = recognizer.detect_signs(vehicle_output)
        else:
            sign_output = vehicle_output

        # Pedestrian risk detection
        final_output, ped_info = ped_detector.detect(sign_output)

        # Persist last detected sign on HUD
        if sign_info["last_sign"]:
            last_sign_label = sign_info["last_sign"]
            last_sign_timer = _scfg["sign_display_frames"]
        elif last_sign_timer > 0:
            last_sign_timer -= 1

        curr_time = time.time()
        display_fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        draw_warnings(final_output, lane_info, vehicle_info, ped_info)
        draw_hud(final_output, lane_info, vehicle_info, sign_info, ped_info,
                 display_fps, paused, last_sign_label, last_sign_timer)

        if writer:
            writer.write(final_output)

        cv2.imshow("ADAS Output", final_output)

    cap.release()
    if writer:
        writer.release()
        print(f"Output saved to: {_vcfg['output_path']}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
