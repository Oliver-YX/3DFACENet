#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch
import face_alignment
from face_alignment import LandmarksType
from pathlib import Path
from skimage.transform import estimate_transform, warp

def bbox2point(left, right, top, bottom, scale=1.25):
    """Compute center and size for similarity transform from bbox or keypoints."""
    old_size = max(right - left, bottom - top)
    center = np.array([(left + right) / 2, (top + bottom) / 2])
    size = old_size * scale
    return size, center

def crop_face_fan(input_folder, output_folder, scale, crop_size, device):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    fa = face_alignment.FaceAlignment(
        LandmarksType._2D, flip_input=False, device=device
    )

    DST_PTS = np.array([
        [0, 0],
        [0, crop_size - 1],
        [crop_size - 1, 0]
    ], dtype=np.float32)

    for fn in tqdm(os.listdir(input_folder), desc="Processing images"):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = Path(input_folder) / fn
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read: {fn}")
            continue

        h, w = img.shape[:2]
        preds = fa.get_landmarks(img)

        if preds is None or len(preds) == 0:
            cropped = cv2.resize(img, (crop_size, crop_size))
        else:
            kpt = np.array(preds[0])
            left, right = kpt[:,0].min(), kpt[:,0].max()
            top, bottom = kpt[:,1].min(), kpt[:,1].max()

            size, center = bbox2point(left, right, top, bottom, scale=scale)
            src_pts = np.array([
                [center[0] - size/2, center[1] - size/2],
                [center[0] - size/2, center[1] + size/2],
                [center[0] + size/2, center[1] - size/2]
            ], dtype=np.float32)

            tform = estimate_transform("similarity", src_pts, DST_PTS)
            img_norm = img.astype(np.float32) / 255.0
            warped = warp(
                img_norm,
                tform.inverse,
                output_shape=(crop_size, crop_size),
                order=1,
                mode="constant",
                cval=0
            )
            cropped = (warped * 255.0).clip(0, 255).astype(np.uint8)

        save_path = output_path / fn
        cv2.imwrite(str(save_path), cropped)

def main():
    parser = argparse.ArgumentParser(
        description="Crop face with FAN + similarity transform; save to input_folder/crop"
    )
    parser.add_argument("--input_folder", "-i", required=True,
                        help="Input image directory (png/jpg)")
    parser.add_argument("--scale", "-s", type=float, default=1.25,
                        help="Face region scale (default: 1.25)")
    parser.add_argument("--crop_size", "-c", type=int, default=224,
                        help="Output crop size (default: 224)")
    parser.add_argument("--device", "-d", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cpu or cuda")

    args = parser.parse_args()
    output_folder = Path(args.input_folder) / "crop"
    crop_face_fan(
        args.input_folder,
        output_folder,
        args.scale,
        args.crop_size,
        args.device
    )

if __name__ == "__main__":
    main()
