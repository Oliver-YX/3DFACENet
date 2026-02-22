#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse

def find_file(folder: str, base: str, exts):
    """Return the first existing path for base+ext in folder."""
    for e in exts:
        p = os.path.join(folder, base + e)
        if os.path.isfile(p):
            return p
    return None

def composite_images(render_folder: str, input_folder: str, mask_folder: str, save_folder: str):
    """
    Occlude images: keep face region, fill background with black.
    Uses input_folder and mask_folder (face=255, bg=0). render_folder kept for API compatibility.
    """
    os.makedirs(save_folder, exist_ok=True)

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG"]
    target_size = (224, 224)

    for fname in sorted(os.listdir(input_folder)):
        base, ext = os.path.splitext(fname)
        path_i = os.path.join(input_folder, fname)
        path_m = find_file(mask_folder, base, exts)

        if not os.path.isfile(path_i):
            print(f"[WARN] Input image not found ({fname}), skip")
            continue
        if path_m is None:
            print(f"[WARN] Mask not found ({base}.*), skip")
            continue

        im_i = cv2.imread(path_i, cv2.IMREAD_COLOR)
        mask = cv2.imread(path_m, cv2.IMREAD_GRAYSCALE)
        if im_i is None or mask is None:
            print(f"[WARN] Cannot read ({base}), skip")
            continue

        im_i = cv2.resize(im_i, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        face_mask = (mask.astype(np.float32) / 255.0)[..., None]
        bg_mask = 1.0 - face_mask

        black_bg = np.zeros_like(im_i, dtype=np.uint8)
        comp = (im_i.astype(np.float32) * face_mask +
                black_bg.astype(np.float32) * bg_mask).astype(np.uint8)

        out_path = os.path.join(save_folder, base + ext)
        cv2.imwrite(out_path, comp)
        print(f"[INFO] Saved face-occluded result: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Keep face region only, fill background with black"
    )
    parser.add_argument("--render_folder", required=False,
                        help="(Unused) placeholder for legacy API")
    parser.add_argument("--input_folder",  required=True,
                        help="Input image folder")
    parser.add_argument("--mask_folder",   required=True,
                        help="Binary mask folder (face=255, bg=0)")
    parser.add_argument("--save_folder",   required=True,
                        help="Output folder for composited images")
    args = parser.parse_args()

    composite_images(
        render_folder=args.render_folder,
        input_folder=args.input_folder,
        mask_folder=args.mask_folder,
        save_folder=args.save_folder
    )

