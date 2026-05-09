import cv2
import numpy as np
import yaml

with open("config.yaml", "r") as f:
    _cfg = yaml.safe_load(f)["lane_detection"]

# Temporal smoothing: store previous frame's lines
_prev_left = None
_prev_right = None

_night_cfg = _cfg["night_mode"]
_clahe = cv2.createCLAHE(
    clipLimit=_night_cfg["clip_limit"],
    tileGridSize=(_night_cfg["tile_grid_size"], _night_cfg["tile_grid_size"])
)

def apply_clahe(frame):
    """Enhance contrast on the L channel in LAB color space."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def is_low_light(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < _night_cfg["brightness_threshold"]

def region_of_interest(image):
    height, width = image.shape[:2]
    roi = _cfg["roi"]

    bottom_y = int(roi["bottom_y"] * height)
    top_y    = int(roi["top_y"] * height)

    polygon = np.array([
        [
            (int(roi["left_bottom_x"] * width),  bottom_y),
            (int(roi["left_top_x"] * width),     top_y),
            (int(roi["right_top_x"] * width),    top_y),
            (int(roi["right_bottom_x"] * width), bottom_y)
        ]
    ], dtype=np.int32)

    mask = np.zeros_like(image)

    if len(image.shape) == 2:
        cv2.fillPoly(mask, polygon, 255)
    else:
        cv2.fillPoly(mask, polygon, (255, 255, 255))

    return cv2.bitwise_and(image, mask)

def select_lane_colors(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    color = _cfg["color"]

    white_mask = cv2.inRange(
        hls,
        np.array(color["white_hls_lower"]),
        np.array(color["white_hls_upper"])
    )
    yellow_mask = cv2.inRange(
        hls,
        np.array(color["yellow_hls_lower"]),
        np.array(color["yellow_hls_upper"])
    )

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(frame, frame, mask=combined_mask)

def make_points(image, line_params):
    slope, intercept = line_params
    height, width = image.shape[:2]

    y1 = int(height * _cfg["roi"]["bottom_y"])
    y2 = int(height * _cfg["roi"]["top_y"])

    if abs(slope) < 0.1:
        slope = 0.1 if slope >= 0 else -0.1

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    left_weights, right_weights = [], []

    if lines is None:
        return None, None

    height, width = image.shape[:2]
    mid_x = width // 2
    slope_min = _cfg["slope_filter"]["min"]
    slope_max = _cfg["slope_filter"]["max"]

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        if x1 == x2:
            continue

        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if abs(slope) < slope_min or abs(slope) > slope_max:
            continue

        # Classify by x-position, not slope sign — more robust for interior cameras
        line_mid_x = (x1 + x2) / 2
        if line_mid_x < mid_x:
            left_fit.append((slope, intercept))
            left_weights.append(length)
        else:
            right_fit.append((slope, intercept))
            right_weights.append(length)

    left_line = None
    right_line = None

    if left_fit:
        left_avg = np.average(left_fit, axis=0, weights=left_weights)
        left_line = make_points(image, left_avg)
        if left_line[0] > mid_x:
            left_line = None

    if right_fit:
        right_avg = np.average(right_fit, axis=0, weights=right_weights)
        right_line = make_points(image, right_avg)
        if right_line[0] < mid_x:
            right_line = None

    return left_line, right_line

def smooth_line(current, previous):
    alpha = _cfg["smoothing_alpha"]
    if previous is None:
        return current
    if current is None:
        return previous
    return (alpha * previous + (1 - alpha) * current).astype(int)

def draw_lane_overlay(frame, left_line, right_line):
    return np.zeros_like(frame)

def detect_lanes(frame):
    """Returns (output_frame, info) where info has offset and departure flag."""
    global _prev_left, _prev_right

    # Night / low-light enhancement
    mode = _night_cfg["enabled"]
    night_active = (mode is True) or (mode == "auto" and is_low_light(frame))
    if night_active:
        frame = apply_clahe(frame)

    color_selected = select_lane_colors(frame)
    gray = cv2.cvtColor(color_selected, cv2.COLOR_BGR2GRAY)
    k = _cfg["blur_kernel"]
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blur, _cfg["canny"]["low_threshold"], _cfg["canny"]["high_threshold"])

    cropped_edges = region_of_interest(edges)

    h = _cfg["hough"]
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=h["rho"],
        theta=np.pi / 180,
        threshold=h["threshold"],
        minLineLength=h["min_line_length"],
        maxLineGap=h["max_line_gap"]
    )

    left_line, right_line = average_slope_intercept(frame, lines)
    left_line = smooth_line(left_line, _prev_left)
    right_line = smooth_line(right_line, _prev_right)
    _prev_left = left_line
    _prev_right = right_line

    overlay = draw_lane_overlay(frame, left_line, right_line)
    output = cv2.addWeighted(frame, 0.8, overlay, 1, 0)

    height, width = frame.shape[:2]
    frame_center = width // 2

    info = {"offset": None, "departure": False, "lines_found": False, "night_mode": night_active}

    if left_line is not None and right_line is not None:
        info["lines_found"] = True
        lane_center = (left_line[0] + right_line[0]) // 2
        offset = frame_center - lane_center
        info["offset"] = offset
        info["departure"] = abs(offset) > _cfg["departure_offset_threshold"]

    return output, info
