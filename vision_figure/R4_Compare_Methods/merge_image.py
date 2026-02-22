# #!/usr/bin/env python3
# import os
# import cv2
# import numpy as np
# import argparse

# def find_file(folder: str, base: str, exts):
#     """
#     在给定文件夹里，用多个后缀匹配同名文件，返回第一个存在的完整路径。
#     """
#     for e in exts:
#         p = os.path.join(folder, base + e)
#         if os.path.isfile(p):
#             return p
#     return None

# def composite_images(render_folder: str, input_folder: str, mask_folder: str, save_folder: str):
#     """
#     对 render_folder 下的渲染图、input_folder 下的原图，以及 mask_folder 下的二值掩码
#     （face=255, bg=0）进行合成：
#       1) 全部图像先调整到 512×512
#       2) final = render * face_mask + original * bg_mask
#     自动匹配 png/jpg/jpeg/bmp 等后缀，无需硬编码扩展名。
#     """
#     os.makedirs(save_folder, exist_ok=True)

#     exts = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG']
#     target_size = (512, 512)

#     for render_fname in sorted(os.listdir(render_folder)):
#         base, rend_ext = os.path.splitext(render_fname)

#         # 找到对应的原始图和 mask
#         path_r = os.path.join(render_folder, render_fname)
#         path_i = find_file(input_folder, base, exts)
#         path_m = find_file(mask_folder,  base, exts)

#         if path_i is None:
#             print(f"[WARN] 原图不存在 ({base}.*)，跳过")
#             continue
#         if path_m is None:
#             print(f"[WARN] 掩码不存在 ({base}.*)，跳过")
#             continue

#         # 读取
#         im_r = cv2.imread(path_r, cv2.IMREAD_COLOR)
#         im_i = cv2.imread(path_i, cv2.IMREAD_COLOR)
#         mask = cv2.imread(path_m, cv2.IMREAD_GRAYSCALE)

#         if im_r is None or im_i is None or mask is None:
#             print(f"[WARN] 无法读取 ({base})，跳过")
#             continue

#         # resize 到 512×512
#         im_r = cv2.resize(im_r, target_size, interpolation=cv2.INTER_LINEAR)
#         im_i = cv2.resize(im_i, target_size, interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

#         # face_mask = 1 where face, bg_mask = 1 where background
#         face_mask = (mask.astype(np.float32) / 255.0)[..., None]
#         bg_mask   = 1.0 - face_mask

#         # 融合：render 用于 face, original 用于 background
#         comp = (im_r.astype(np.float32) * face_mask +
#                 im_i.astype(np.float32) * bg_mask).astype(np.uint8)

#         # 保存
#         out_path = os.path.join(save_folder, base + rend_ext)
#         cv2.imwrite(out_path, comp)
#         print(f"[INFO] 合成并保存: {out_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="合成渲染图与原图：自动匹配扩展名，无需硬编码"
#     )
#     parser.add_argument("--render_folder", required=True,
#                         help="渲染图所在文件夹")
#     parser.add_argument("--input_folder",  required=True,
#                         help="原始输入图像文件夹")
#     parser.add_argument("--mask_folder",   required=True,
#                         help="二值掩码文件夹（face=255, bg=0）")
#     parser.add_argument("--save_folder",   required=True,
#                         help="合成结果保存文件夹")
#     args = parser.parse_args()

#     composite_images(
#         render_folder=args.render_folder,
#         input_folder=args.input_folder,
#         mask_folder=args.mask_folder,
#         save_folder=args.save_folder
#     )


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
    Uses input_folder and mask_folder (face=255, bg=0). render_folder kept for compatibility.
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

