#!/usr/bin/env python3
"""
Generate and save full‐image face masks using MediaPipe Face Mesh—no cropping or resizing.

For each image in --input_folder:
  1) Detect the 468‐point face mesh.
  2) Compute the convex hull of all landmarks → binary mask of same size as input.
  3) Save mask (face=255, background=0) to --output_folder with same basename.

Usage:
  pip install mediapipe opencv-python tqdm
  python save_face_masks.py \
    --input_folder  path/to/images \
    --output_folder path/to/masks
"""

import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import mediapipe as mp

mp_face = mp.solutions.face_mesh

def process_one(image_path: str, output_folder: str):
    name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] cannot read {image_path}")
        return

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3
    ) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
        else:
            # no face detected → leave mask all‐zeros
            print(f"[WARN] no face detected in {name}")

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{name}.png")
    cv2.imwrite(out_path, mask)
    
def main():
    parser = argparse.ArgumentParser(description="Save full‐size face masks via MediaPipe Face Mesh")
    parser.add_argument("--input_folder",  required=True, help="Input image folder")
    parser.add_argument("--output_folder", required=True, help="Output folder for masks")
    args = parser.parse_args()

    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    paths = []
    for e in exts:
        paths += glob(os.path.join(args.input_folder, e))
    paths = sorted(paths)

    for p in tqdm(paths):
        process_one(p, args.output_folder)

if __name__ == "__main__":
    main()
