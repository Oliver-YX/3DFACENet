# 3DFACENet

This repository contains the implementation of **3DFACENet**, for 3D facial attractiveness computation and enhancement.

The method supports both [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) and [FLAME](https://flame.is.tue.mpg.de/) for 3D face representation.

---

## Environment

- **Python**: 3.8 or 3.9 recommended.
- **CUDA**: Required for GPU training and rendering; version should match your PyTorch/CUDA stack.
- **Dependencies**: Follow the environment setup of either:
  - **[Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)** — for BFM2009 reconstruction, rendering (nvdiffrast), and coefficient-based pipelines.
  - **[AlbedoGAN / Towards-Realistic-Generative-3D-Face-Models](https://github.com/aashishrai3799/Towards-Realistic-Generative-3D-Face-Models)** — for FLAME-based pipelines (DECA/MICA, etc.).

### Required: merge external repos into this project

Before running the BFM or FLAME pipelines, you must place the full contents of the dependency repos under the corresponding folders:

- **Deep3D**: Clone or download [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch), then **move all of its files and folders** into **`./bfm_transform`** (so that Deep3D’s `models`, `util`, `nvdiffrast`, etc. sit inside `bfm_transform/`).
- **AlbedoGAN**: Clone or download [Towards-Realistic-Generative-3D-Face-Models](https://github.com/aashishrai3799/Towards-Realistic-Generative-3D-Face-Models), then **place all of its files and folders** into **`./flame_transform`** (so that AlbedoGAN’s code and assets sit inside `flame_transform/`).

This repo only contains the 3DFACENet entry scripts in `bfm_transform/` and `flame_transform/`; the subfolders (from Deep3D and AlbedoGAN) are not included and must be added locally as above.

- BFM assets: download **01_MorphableModel.mat** from the [BFM page](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads) and **Exp_Pca.bin** from [Guo et al.](https://github.com/Juyong/3DFace) ([direct link](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing)), and organize as:

```
3DFACENet
└── BFM
    ├── 01_MorphableModel.mat
    ├── Exp_Pca.bin
    └── ...
```

---

## Path configuration (required before run)

Scripts use placeholder paths (e.g. `your_location`, `your_save_path`). **Replace these with your actual paths** before running.

| Script / module | Variable / location | Meaning |
|-----------------|---------------------|--------|
| **bfm_coefficient/prepare_coef_score_excel.py** | `save_path` | Directory to save `train_coef.xlsx`, `test_coef.xlsx`, `all_coef.xlsx`. |
| | `directory` | Directory containing 3D reconstruction `.mat` results (one `.mat` per image). |
| | `train_file`, `test_file` | Paths to SCUT-FBP5500 train/test split files (e.g. `train.txt`, `test.txt`). |
| **train_prediction_model.py** | `save_path` | Where to save trained model and `flame_trainloss.txt`. |
| | `directory` | Folder of reconstruction `.mat` files. |
| | `YOUR_COEF_ROOT` | Folder containing `train_coef.xlsx`, `test_coef.xlsx`, `all_coef.xlsx` (if you already have them). |
| **train_prediction_model_5folds.py** | `save_path` | Output directory for fold models and `trainloss.txt`. |
| | `directory` | Folder of `.mat` files for 5-fold training. |
| | `cross_val_base_dir` | SCUT-FBP5500 5-fold split root (e.g. `.../5_folders_cross_validations_files`). |
| **bfm_transform/transfor2mesh_all.py** | `all_file_path` | Path to coefficient Excel (e.g. `bfm_coefficient/all_coef.xlsx`) or set to `your_location/all_coef.xlsx`. |
| **bfm_transform/transfor2mesh_all.py** (main block) | `folder_path` | Input folder with images (and optional `mat/`, `detections/`). |
| | `save_folder` | Output folder for beautified meshes/images. |
| **bfm_transform/transfor2mesh_all_auto_k.py** | `all_file_path` | Coefficient Excel path. |
| | `folder_path`, `save_folder` | Input image folder and output folder for beautified results. |
| **bfm_transform/coefficient_rating.py** | `path` | Directory with `optimized_svr_model_*.pkl` and `scaler_*.pkl`. |
| | `save_path` | Folder containing `modify_*_*.txt` (e.g. beautify_demo). |
| **bfm_transform/render_image.py** | `all_file_path`, `mat_input_path` | Coefficient Excel and `.mat` results directory. |
| | `image_folder`, `detections_folder`, `save_folder` | Input images, landmark detections, and render output. |
| **bfm_transform/render_image_auto_k.py** | Same as above + `all_file_path`, `mat_input_path`. |
| **bfm_transform/render_coeff_to_image.py** | `save_folder` | Output directory for rendered images and meshes. |
| **bfm_transform/average_improve_compute.py** | `all_file_path`, `mat_input_path` | Coefficient Excel and `.mat` directory; optional `save_file_path` for Excel output. |
| **flame_transform/transfor_2_mesh.py** | `all_file_path` | FLAME coefficient Excel path. |
| **flame_transform/img_2_tex.py** | `savefolder`, `inputpath` | Output and input directories for texture extraction. |
| **bfm_coefficient/visualizaiton/** (attractive_center_clustering*.py) | `YOUR_COEF_ROOT` | Folder containing `all_coef.xlsx`. |
| | `YOUR_VIZ_SAVE_PATH` | Where to save clustering/visualization figures. |
| **vis_network.py** | `YOUR_MODEL_ROOT` | Directory with `alexnet.pth`, `resnet18.pth`, `HMTNet.pth`. |
| **param_compute.py** | `model_path` | Path to saved SVR `.pkl` model. |

---

## Controlling 3D face attractiveness enhancement

- **BFM**: In `bfm_transform/transfor2mesh_all.py`, set `all_file_path`, `folder_path`, and `save_folder`, then run. You can choose a beautification level or edit shape/texture coefficients. Attractive centers are in `bfm_transform/F_center_new.txt` and `bfm_transform/M_center_new.txt`.
- **FLAME**: Use `flame_transform/transfor_2_mesh.py` (set `all_file_path` and run the beautification/export steps as in the script).

### BFM workflow (example)

1. Open `bfm_transform/transfor2mesh_all.py`.
2. Set `all_file_path` to your coefficient Excel; set `folder_path` (input images) and `save_folder` (output).
3. Run the script to get beautified meshes and rendered images.

---

## Accurate 3D face attractiveness computation

After 3D reconstruction you obtain `.mat` files with BFM or FLAME coefficients. Use them to train and run the attractiveness scorer.

### Training

- **Single split (e.g. 60/40)**: `train_prediction_model.py` — set `save_path`, `directory` (or `YOUR_COEF_ROOT` if using pre-built Excel), and run.
- **5-fold cross-validation**: `train_prediction_model_5folds.py` — set `save_path`, `directory`, and `cross_val_base_dir` (SCUT-FBP5500 5-fold splits), then run.

Both scripts expect [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) train/test split files and coefficient data (from `.mat` or from Excel produced by `bfm_coefficient/prepare_coef_score_excel.py`).

### Scoring coefficients

- Use `bfm_transform/coefficient_rating.py`: set `path` to the folder with trained `optimized_svr_model_*.pkl` and `scaler_*.pkl`, and `save_path` to the folder containing the `modify_*_*.txt` coefficient file to score.

---

## Citation

If you use this code or our findings, please cite:

```bibtex
@article{xie20253dfacenet,
  title   = {3DFACENet: 3D Facial Attractiveness Computation and Enhancement Network},
  author  = {Xie, Yuan and Peng, Tianhao and Li, Mu and Wu, Baoyuan and Zhang, David},
  journal = {IEEE Transactions on Image Processing},
  year    = {2025},
  publisher = {IEEE}
}
```

Thank you for your interest in 3DFACENet.
