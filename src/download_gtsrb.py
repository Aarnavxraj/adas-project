import os
import sys
import zipfile
import csv
import shutil
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Paths (run from project root via:  .venv/bin/python3 src/download_gtsrb.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_TRAIN    = os.path.join(PROJECT_ROOT, "data", "traffic_signs", "train")
OUT_TEST     = os.path.join(PROJECT_ROOT, "data", "traffic_signs", "test")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "data", "_gtsrb_cache")

# Official GTSRB mirror (Universität Bochum / erda.dk)
BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
TRAIN_ZIP = "GTSRB_Final_Training_Images.zip"
TEST_IMG_ZIP = "GTSRB_Final_Test_Images.zip"
TEST_GT_ZIP  = "GTSRB_Final_Test_GT.zip"

# 43 GTSRB class names (index = class id)
CLASS_NAMES = [
    "speed_limit_20", "speed_limit_30", "speed_limit_50", "speed_limit_60",
    "speed_limit_70", "speed_limit_80", "end_speed_limit_80", "speed_limit_100",
    "speed_limit_120", "no_overtaking", "no_overtaking_trucks", "priority_road_next",
    "priority_road", "give_way", "stop", "no_vehicles", "no_trucks", "no_entry",
    "general_caution", "dangerous_curve_left", "dangerous_curve_right",
    "double_curve", "bumpy_road", "slippery_road", "road_narrows_right",
    "road_work", "traffic_signals", "pedestrians", "children_crossing",
    "bicycles_crossing", "ice_snow", "wild_animals", "end_all_restrictions",
    "turn_right_ahead", "turn_left_ahead", "ahead_only", "go_straight_or_right",
    "go_straight_or_left", "keep_right", "keep_left", "roundabout",
    "end_no_overtaking", "end_no_overtaking_trucks"
]


def _download(url, dest):
    if os.path.exists(dest):
        print(f"  Already cached: {os.path.basename(dest)}")
        return
    print(f"  Downloading {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    print(f"  Done: {os.path.basename(dest)}")


def _make_class_dirs():
    for name in CLASS_NAMES:
        os.makedirs(os.path.join(OUT_TRAIN, name), exist_ok=True)
        os.makedirs(os.path.join(OUT_TEST,  name), exist_ok=True)


def organize_train(zip_path):
    """
    Training zip structure:
      GTSRB/Final_Training/Images/<NNNNN>/*.ppm
    Each class folder is named with its zero-padded 5-digit class id.
    """
    print("Extracting training images ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".ppm")]
        total = len(members)
        for i, member in enumerate(members):
            parts = member.split("/")
            # parts[-2] is the class id folder, parts[-1] is the filename
            class_id = int(parts[-2])
            class_name = CLASS_NAMES[class_id]
            dst_folder = os.path.join(OUT_TRAIN, class_name)
            fname = os.path.basename(member)
            dst_path = os.path.join(dst_folder, fname)
            if not os.path.exists(dst_path):
                data = zf.read(member)
                with open(dst_path, "wb") as f:
                    f.write(data)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{total} training images done ...")
    print(f"  Training images organised ({total} files).")


def organize_test(img_zip_path, gt_zip_path):
    """
    Test zip structure:
      GTSRB/Final_Test/Images/*.ppm
    Labels are in GT-final_test.csv inside the GT zip.
    CSV columns: Filename;Width;Height;Roi.X1;...;ClassId
    """
    # Extract GT CSV
    print("Extracting test labels ...")
    gt_csv_path = os.path.join(CACHE_DIR, "GT-final_test.csv")
    if not os.path.exists(gt_csv_path):
        with zipfile.ZipFile(gt_zip_path, "r") as zf:
            csv_member = [m for m in zf.namelist() if m.endswith(".csv")][0]
            data = zf.read(csv_member)
            with open(gt_csv_path, "wb") as f:
                f.write(data)

    # Build filename -> class_id map
    label_map = {}
    with open(gt_csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            label_map[row["Filename"]] = int(row["ClassId"])

    print("Extracting test images ...")
    with zipfile.ZipFile(img_zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".ppm")]
        total = len(members)
        for i, member in enumerate(members):
            fname = os.path.basename(member)
            class_id = label_map.get(fname)
            if class_id is None:
                continue
            class_name = CLASS_NAMES[class_id]
            dst_folder = os.path.join(OUT_TEST, class_name)
            dst_path = os.path.join(dst_folder, fname)
            if not os.path.exists(dst_path):
                data = zf.read(member)
                with open(dst_path, "wb") as f:
                    f.write(data)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{total} test images done ...")
    print(f"  Test images organised ({total} files).")


def count_images(root):
    n = 0
    for dirpath, _, files in os.walk(root):
        n += sum(1 for f in files if f.lower().endswith((".ppm", ".png", ".jpg")))
    return n


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    _make_class_dirs()

    train_zip = os.path.join(CACHE_DIR, TRAIN_ZIP)
    test_img_zip = os.path.join(CACHE_DIR, TEST_IMG_ZIP)
    test_gt_zip  = os.path.join(CACHE_DIR, TEST_GT_ZIP)

    print("=== Downloading GTSRB ===")
    _download(f"{BASE_URL}/{TRAIN_ZIP}",   train_zip)
    _download(f"{BASE_URL}/{TEST_IMG_ZIP}", test_img_zip)
    _download(f"{BASE_URL}/{TEST_GT_ZIP}",  test_gt_zip)

    print("\n=== Organising Training Set ===")
    organize_train(train_zip)

    print("\n=== Organising Test Set ===")
    organize_test(test_img_zip, test_gt_zip)

    n_train = count_images(OUT_TRAIN)
    n_test  = count_images(OUT_TEST)
    print(f"\nAll done!")
    print(f"  Train images : {n_train}  →  {OUT_TRAIN}")
    print(f"  Test  images : {n_test}   →  {OUT_TEST}")
    print("\nYou can now run:  .venv/bin/python3 src/train_sign_model.py")


if __name__ == "__main__":
    main()
