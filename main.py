import os
import cv2
import numpy as np

from Yolov5.v5 import(
    get_image_files,
    load_model as load_v5,
    run_image as run_v5_image,
)
from Yolov8.v8 import(
    load_model as load_v8,
    run_image as run_v8_image,
)

INPUT_DIR = "Images_to_test"
RESIZED_DIR = "Resized_Images"
OUTPUT_DIR_V5 = "Images_that_testedv5"
OUTPUT_DIR_V8 = "Images_that_testedv8"
CONF_THRESH = 0.5
YOLOV5_MODEL ="yolov5n"
YOLOV8_MODEL ="yolov8x.pt"
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 640
KEEP_ASPECT = True

def clear_folder(folder: str):

    os.makedirs(folder, exist_ok = True)
    for fn in os.listdir(folder):
        full_path = os.path.join(folder,fn)
        if os.path.isfile(full_path):
            os.remove(full_path)

def get_image_files(folder: str):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return[
        fn 
        for fn in os.listdir(folder)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

def resize_folder_images(
        input_folder: str,
        output_folder: str,
        new_width: int,
        new_height:int,
        keep_aspect: bool = True
):
    
    os.makedirs(output_folder, exist_ok = True)

    for fname in get_image_files(input_folder):
        src_path = os.path.join(input_folder, fname)
        img = cv2.imread(src_path)
        if img is None:
            print(f"{fname} (could not read)")
            continue
        
        h,w = img.shape[:2]

        if keep_aspect:
            scale = min(new_width / w, new_height / h)
            resized_w = int(w*scale)
            resized_h = int(h*scale)

            img_resized = cv2.resize(
                img, (resized_w, resized_h), interpolation = cv2.INTER_AREA
            )
            
            canvas = 255 * np.ones((new_height, new_width, 3), dtype = np.uint8)

            x_offset = (new_width - resized_w) // 2
            y_offset = (new_height - resized_h) // 2
            canvas[y_offset: y_offset + resized_h,
                   x_offset: x_offset + resized_w] = img_resized
            
            out_img = canvas
        else:
            out_img = cv2.resize(
                img, (new_width, new_height), interpolation = cv2.INTER_AREA
            )
        
        dst_path = os.path.join(output_folder, fname)
        cv2.imwrite(dst_path, out_img)
        print(f"Resized {fname} {new_width}x{new_height} (saved to {output_folder}/{fname})")

def main():

    print(f"\n>>> Resizing images from '{INPUT_DIR}'  '{RESIZED_DIR}' at {RESIZE_WIDTH}×{RESIZE_HEIGHT}\n")          
    if os.path.isdir(RESIZED_DIR):
        clear_folder(RESIZED_DIR)
    else:
        os.makedirs(RESIZED_DIR, exist_ok = True)
    

    resize_folder_images(
        input_folder = INPUT_DIR,
        output_folder = RESIZED_DIR,
        new_width = RESIZE_WIDTH,
        new_height = RESIZE_HEIGHT,
        keep_aspect = KEEP_ASPECT
    )

    clear_folder(OUTPUT_DIR_V5)
    clear_folder(OUTPUT_DIR_V8)

    print("\n>>> Loading YOLO model...\n")
    v5_model = load_v5(YOLOV5_MODEL)
    v8_model = load_v8(YOLOV8_MODEL)

    print("\n>>> Running YOLOv5/v8 on resized images:\n")
    for fname in get_image_files(RESIZED_DIR):
        src_resized = os.path.join(RESIZED_DIR, fname)

        ann_v5 = run_v5_image(v5_model, src_resized, CONF_THRESH )

        if ann_v5 is not None:

            out_v5_path = os.path.join(OUTPUT_DIR_V5, f"v5_detected_{fname}")
            cv2.imwrite(out_v5_path, ann_v5)
            print(f"{fname} YOLOv5 (saved to {out_v5_path})")
        else:

            ann_v8 = run_v8_image(v8_model, src_resized)
            out_v8_path = os.path.join(OUTPUT_DIR_V8, f"v8_detected_{fname}")
            cv2.imwrite(out_v8_path, ann_v8)
            print(f"{fname} YOLOv8 (saved to {out_v8_path})")
    
    print("\n>>> ALL done. Check the output folders:")
    print(f"    • YOLOv5 results: {OUTPUT_DIR_V5}")
    print(f"    • YOLOv8 results: {OUTPUT_DIR_V8}\n")

if __name__ == "__main__":
    main()
  



# INPUT_DIR = "Images_to_Test"
# OUTPUT_DIR_V5 = "Images_that_testedv5"
# OUTPUT_DIR_V8 = "Images_that_testedv8"
# CONF_THRESH = 0.5

# def clear_folder(folder: str):

#     os.makedirs(folder, exist_ok = True)
#     for fn in os.listdir(folder):
#         path = os.path.join(folder, fn)
#         if os.path.isfile(path):
#             os.remove(path)

# def main():
#     clear_folder(OUTPUT_DIR_V5)
#     clear_folder(OUTPUT_DIR_V8)
#     v5 = load_v5("yolov5n")
#     v8 = load_v8("yolov8x.pt")

#     for fname in get_image_files(INPUT_DIR):
#         src = os.path.join(INPUT_DIR, fname)

#         ann = run_v5_image(v5, src, CONF_THRESH)
#         if ann is not None:
#             dst = os.path.join(OUTPUT_DIR_V5, f"v5_detected_{fname}")
#             tag = "v5"
        
#         else:
#             ann = run_v8_image(v8, src)
#             dst = os.path.join(OUTPUT_DIR_V8, f"v8_detected_{fname}")
#             tag = "v8"
        
#         cv2.imwrite(dst,ann)
#         print(f"{fname} {tag} {dst}")

# if __name__ == "__main__":
#     main()