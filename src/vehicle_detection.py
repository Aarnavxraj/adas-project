import cv2
import yaml
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

with open("config.yaml", "r") as f:
    _cfg = yaml.safe_load(f)["vehicle_detection"]

_tcfg = _cfg["tracker"]


class CentroidTracker:
    """Assigns persistent IDs to detections and estimates speed."""

    def __init__(self, fps):
        self.next_id = 0
        self.tracks = {}  # id -> {centroid, lost, speed_history}
        self.fps = fps

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2, class_name, conf)
        Returns: list of (x1, y1, x2, y2, class_name, conf, track_id, speed_kmh)
        """
        max_dist = _tcfg["max_centroid_distance"]
        max_lost = _tcfg["max_lost_frames"]
        ppm = _tcfg["pixels_per_meter"]
        smooth = _tcfg["speed_smoothing"]

        new_centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2, *_ in detections]

        # Match detections to existing tracks by nearest centroid
        used_track_ids = set()
        used_det_indices = set()
        matches = {}  # det_index -> track_id

        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            track_centroids = [self.tracks[tid]["centroid"] for tid in track_ids]

            for di, dc in enumerate(new_centroids):
                best_dist = float("inf")
                best_tid = None
                for ti, tid in enumerate(track_ids):
                    if tid in used_track_ids:
                        continue
                    tc = track_centroids[ti]
                    dist = np.hypot(dc[0] - tc[0], dc[1] - tc[1])
                    if dist < best_dist and dist < max_dist:
                        best_dist = dist
                        best_tid = tid
                if best_tid is not None:
                    matches[di] = best_tid
                    used_track_ids.add(best_tid)
                    used_det_indices.add(di)

        # Update matched tracks
        for di, tid in matches.items():
            prev_centroid = self.tracks[tid]["centroid"]
            curr_centroid = new_centroids[di]
            pixel_disp = np.hypot(
                curr_centroid[0] - prev_centroid[0],
                curr_centroid[1] - prev_centroid[1]
            )
            speed_kmh = (pixel_disp / ppm) * self.fps * 3.6
            history = self.tracks[tid]["speed_history"]
            history.append(speed_kmh)
            if len(history) > smooth:
                history.pop(0)
            self.tracks[tid]["centroid"] = curr_centroid
            self.tracks[tid]["lost"] = 0

        # Register new tracks for unmatched detections
        for di in range(len(detections)):
            if di not in used_det_indices:
                self.tracks[self.next_id] = {
                    "centroid": new_centroids[di],
                    "lost": 0,
                    "speed_history": [0.0]
                }
                matches[di] = self.next_id
                self.next_id += 1

        # Age unmatched tracks and remove stale ones
        for tid in list(self.tracks.keys()):
            if tid not in used_track_ids and tid not in matches.values():
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > max_lost:
                    del self.tracks[tid]

        # Build output
        results = []
        for di, det in enumerate(detections):
            x1, y1, x2, y2, class_name, conf = det
            tid = matches.get(di)
            if tid is not None and tid in self.tracks:
                history = self.tracks[tid]["speed_history"]
                speed_kmh = sum(history) / len(history)
            else:
                speed_kmh = 0.0
            results.append((x1, y1, x2, y2, class_name, conf, tid, speed_kmh))

        return results


class VehicleDetector:
    def __init__(self, fps=30):
        self.model = YOLO(_cfg["model_path"])
        self.conf_threshold = _cfg["conf_threshold"]
        self.vehicle_classes = set(_cfg["vehicle_classes"])
        self.tracker = CentroidTracker(fps)

    def detect_vehicles(self, frame):
        """Returns (output_frame, info) where info has vehicle_ahead, fcw_level, max_speed_kmh."""
        output = frame.copy()
        results = self.model(frame, verbose=False)[0]

        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_area = frame_width * frame_height

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = self.model.names[cls_id]

            if conf < self.conf_threshold:
                continue
            if class_name not in self.vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append((x1, y1, x2, y2, class_name, conf))

        tracked = self.tracker.update(detections)

        fcw_level = "none"
        max_speed = 0.0

        for x1, y1, x2, y2, class_name, conf, tid, speed_kmh in tracked:
            box_area = (x2 - x1) * (y2 - y1)
            box_center_x = (x1 + x2) // 2
            area_ratio = box_area / frame_area
            is_centered = abs(box_center_x - frame_center_x) < frame_width * _cfg["ahead_center_proximity"]

            if is_centered:
                if area_ratio > _cfg["fcw_critical_ratio"]:
                    fcw_level = "critical"
                elif area_ratio > _cfg["ahead_area_ratio"] and fcw_level != "critical":
                    fcw_level = "warning"

            if is_centered and area_ratio > _cfg["ahead_area_ratio"]:
                color = (0, 0, 255) if fcw_level == "critical" else (0, 165, 255)
            else:
                color = (0, 255, 0)

            if speed_kmh > max_speed:
                max_speed = speed_kmh

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f} | {speed_kmh:.0f} km/h"
            cv2.putText(output, label, (x1, max(y1 - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        info = {
            "vehicle_ahead": fcw_level != "none",
            "fcw_level": fcw_level,
            "max_speed_kmh": max_speed
        }

        return output, info
