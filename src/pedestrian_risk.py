import cv2
import yaml
import numpy as np
from ultralytics import YOLO

with open("config.yaml", "r") as f:
    _cfg = yaml.safe_load(f)["pedestrian_risk"]


class PedestrianRiskDetector:
    """Detects pedestrians and estimates proximity risk level."""

    def __init__(self):
        self.model = YOLO(_cfg["model_path"])
        self.conf_threshold = _cfg["conf_threshold"]
        # Thresholds as fraction of frame area occupied by bounding box
        self.danger_ratio  = _cfg["danger_area_ratio"]   # e.g. 0.04  -> very close
        self.caution_ratio = _cfg["caution_area_ratio"]  # e.g. 0.015 -> approaching

    def detect(self, frame):
        """
        Run pedestrian detection on frame.

        Returns
        -------
        output : np.ndarray   annotated frame
        info   : dict         {pedestrians: list[dict], risk_level: str}
                              risk_level in {"none", "caution", "danger"}
        """
        output = frame.copy()
        results = self.model(frame, verbose=False)[0]

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h

        pedestrians = []
        highest_risk = "none"

        for box in results.boxes:
            cls_id     = int(box.cls[0].item())
            conf       = float(box.conf[0].item())
            class_name = self.model.names[cls_id]

            if class_name != "person" or conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # --- False-positive filters ---

            # 1. Ignore detections whose TOP edge starts in the bottom 55% of
            #    the frame — those are almost always the driver's hands/body.
            if y1 > frame_h * 0.55:
                continue

            # 2. Ignore detections whose BOTTOM edge sits below 90% of the
            #    frame height and whose top is also low — dashboard / interior.
            if y2 > frame_h * 0.90 and y1 > frame_h * 0.40:
                continue

            # 3. Real pedestrians are taller than wide. Skip squat blobs
            #    (hands, steering wheel, rear-view mirror reflections).
            box_w = x2 - x1
            box_h = y2 - y1
            if box_h == 0 or (box_w / box_h) > 1.4:
                continue

            box_area   = box_w * box_h
            area_ratio = box_area / frame_area

            # Risk level for this pedestrian
            if area_ratio >= self.danger_ratio:
                risk = "danger"
            elif area_ratio >= self.caution_ratio:
                risk = "caution"
            else:
                risk = "none"

            # Upgrade global risk
            if risk == "danger":
                highest_risk = "danger"
            elif risk == "caution" and highest_risk == "none":
                highest_risk = "caution"

            pedestrians.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "risk": risk
            })

            # Draw bounding box
            if risk == "danger":
                color = (0, 0, 255)    # red
                label = f"PEDESTRIAN DANGER {conf:.2f}"
            elif risk == "caution":
                color = (0, 165, 255)  # orange
                label = f"Pedestrian Caution {conf:.2f}"
            else:
                color = (0, 255, 200)  # teal
                label = f"Pedestrian {conf:.2f}"

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        info = {
            "pedestrians": pedestrians,
            "risk_level":  highest_risk
        }
        return output, info
