import os
import cv2
import torch

def get_image_files(folder: str):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return[
        fn for fn in os.listdir(folder)
        if fn.lower().endswith(('.png','.jpg', '.jpeg'))
    ]

def load_model(model_name: str = "yolov5n"):

    return torch.hub.load("ultralytics/yolov5", model_name, pretrained = True)

def detect_and_annotate(model, img_path: str):
    results = model(img_path)
    results.render()
    return results.ims[0]

def run_image(model, img_path: str, conf_thres: float):
    results = model(img_path)
    dets = results.xyxy[0].cpu().numpy()
    if dets.size and dets[:,4].max() >= conf_thres:
        results.render()
        return results.ims[0]
    return None

def clear_folder(folder: str):
    if not os.path.isdir(folder):
        return
    for fn in os.listdir(folder):
        path = os.path.join(folder, fn)
        if os.path.isfile(path):
            os.remove(path)

def run_batch(
        input_folder: str = "Images_to_test",
        output_folder: str = "Images_that_testedv5",
        model_name: str = "yolov5n"
):
    os.makedirs(output_folder, exist_ok = True)
    clear_folder(output_folder)

    model = load_model(model_name)
    for fname in get_image_files(input_folder):
        src = os.path.join(input_folder, fname)
        out_name = f"v5_detected_{fname}"
        dst = os.path.join(output_folder, out_name)

        ann = detect_and_annotate(model, src)
        cv2.imwrite(dst, ann)
        print(f"v5: {fname} {out_name}")

if __name__ == "__main__":
    run_batch()