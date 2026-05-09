import cv2
import os
import numpy as np
import yaml
from ultralytics import YOLO

with open("config.yaml", "r") as f:
    _cfg = yaml.safe_load(f)["sign_recognition"]

# COCO classes that are traffic signs/lights
_YOLO_SIGN_CLASSES = {"stop sign", "traffic light"}

class TrafficSignRecognizer:
    def __init__(self):
        self.model = YOLO(_cfg["yolo_model_path"])
        self.conf_threshold = _cfg["conf_threshold"]

        # Load custom CNN if the trained model exists
        self.custom_model = None
        self.custom_class_names = []
        custom_path = _cfg["custom_model_path"]
        if os.path.exists(custom_path):
            import tensorflow as tf
            self.custom_model = tf.keras.models.load_model(custom_path)
            # Try to load class names from training directory
            train_dir = yaml.safe_load(open("config.yaml"))["training"]["train_dir"]
            if os.path.exists(train_dir):
                self.custom_class_names = sorted(os.listdir(train_dir))
            print(f"Custom sign model loaded: {custom_path}")
        else:
            print("No custom sign model found — using YOLO sign detection only.")

    def _classify_roi(self, roi):
        """Run custom CNN on a cropped sign region."""
        img = cv2.resize(roi, (32, 32)).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        preds = self.custom_model.predict(img, verbose=0)[0]
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        label = self.custom_class_names[class_id] if class_id < len(self.custom_class_names) else str(class_id)
        return label, confidence

    def detect_signs(self, frame):
        """Returns (output_frame, info) where info has detected signs list and last_sign."""
        output = frame.copy()
        results = self.model(frame, verbose=False)[0]

        detected = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = self.model.names[cls_id]

            if conf < self.conf_threshold:
                continue
            if class_name not in _YOLO_SIGN_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = class_name

            # Optionally refine with custom model
            if self.custom_model is not None:
                roi = frame[max(y1, 0):y2, max(x1, 0):x2]
                if roi.size > 0:
                    custom_label, custom_conf = self._classify_roi(roi)
                    if custom_conf > 0.6:
                        label = custom_label

            detected.append({"label": label, "conf": conf, "bbox": (x1, y1, x2, y2)})

            # Draw blue box for signs
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(output, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        info = {
            "signs": detected,
            "last_sign": detected[0]["label"] if detected else None
        }

        return output, info
