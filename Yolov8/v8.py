import os
import cv2
from ultralytics import YOLO

def get_image_files(folder: str):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return[
        fn for fn in os.listdir(folder)
        if fn.lower().endswith(('.png','.jpg', '.jpeg'))
    ]

def load_model(weights: str = "yolov8x.pt"):
    return YOLO(weights)

def detect_and_annotate(model, img_path: str):
    results = model(img_path)
    return results[0].plot()

def run_image(model, img_path: str):
    return detect_and_annotate(model, img_path)

def clear_folder(folder: str):
    if not os.path.isdir(folder):
        return
    for fn in os.listdir(folder):
        path = os.path.join(folder,fn)
        if os.path.isfile(path):
            os.remove(path)

def run_batch(
        input_folder: str = "Images_to_test",
        output_folder: str = "Images_that_testedv8",
        weights: str = "yolov8x.pt",
):
    os.makedirs(output_folder, exist_ok = True)
    clear_folder(output_folder)
    
    model = load_model(weights)
    for fname in get_image_files(input_folder):
        src = os.path.join(input_folder, fname)
        out_name = f"v8_detected_{fname}"
        dst = os.path.join(output_folder, out_name)

        ann = run_image(model, src)
        cv2.imwrite(dst, ann)
        print(f"v8: {fname} {out_name}")

if __name__ == "__main__":
    run_batch()