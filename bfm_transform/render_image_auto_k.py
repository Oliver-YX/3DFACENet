import os
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import pickle
from options.test_options import TestOptions
from models import create_model
from util.visualizer import MyVisualizer
from util.nvdiffrast import MeshRenderer
from util.preprocess import align_img
from test import read_data
from PIL import Image
import torch
from torchvision import transforms
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
from util.load_mats import load_lm3d
from util.load_mats_full import load_lm3d_full
from util.util import tensor2im, save_image
import trimesh
from models.bfm import ParametricFaceModel
from models.base_model import BaseModel
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)
import concurrent.futures  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

all_file_path = "your_location/all_coef.xlsx"
mat_input_path = "your_location/mat"

# Read score table for auto-k mode
score_df = pd.read_excel(all_file_path)

# adaptive k calculation function
def adaptive_k(score, distance, alpha=5):
    eps = 1e-6
    return np.exp(-alpha * score / ((distance + eps)**2)) * 100

# initialize model and renderer
# facemodel = ParametricFaceModel(
#     bfm_folder="BFM", camera_distance=10., focal=1015., center=112.,
#     is_train=False, default_name='BFM_model_front.mat'
# ).to(device)
# # facemodel.face_buf = facemodel.face_buf.to(device)
facemodel = ParametricFaceModel(
    bfm_folder="BFM", camera_distance=10., focal=1015., center=112.,
    is_train=False, default_name='BFM_model_front.mat'
)

facemodel.to(device)
facemodel.face_buf = facemodel.face_buf.to(device)

fov = 2 * np.arctan(112.0 / 1015.0) * 180 / np.pi
cameras = FoVPerspectiveCameras(fov=fov, znear=5.0, zfar=15.0)
raster_settings = RasterizationSettings(
    image_size=int(2 * 112.0),
    blur_radius=0.0,
    faces_per_pixel=1,
    bin_size=None,
)
renderer = MeshRenderer(
    rasterize_fov=fov, znear=5.0, zfar=15.0, rasterize_size=int(2 * 112.0)
)
mesh_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
lm3d_std = load_lm3d("BFM")

def load_vectors_from_mat(name, directory):
    file_path = os.path.join(directory, name)
    mat_data = loadmat(file_path)
    return {
        "id":    mat_data["id"].flatten(),
        "exp":   mat_data["exp"].flatten(),
        "tex":   mat_data["tex"].flatten(),
        "angle": mat_data["angle"].flatten(),
        "gamma": mat_data["gamma"].flatten(),
        "trans": mat_data["trans"].flatten()
    }

def generate_coeff(id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations):
    id_t = torch.from_numpy(np.array(id_coeffs))
    exp_t = torch.from_numpy(np.array(exp_coeffs))
    tex_t = torch.from_numpy(np.array(tex_coeffs))
    ang_t = torch.from_numpy(np.array(angles))
    gam_t = torch.from_numpy(np.array(gammas))
    trn_t = torch.from_numpy(np.array(translations))
    coeffs = torch.cat((id_t, exp_t, tex_t, ang_t, gam_t, trn_t), dim=0)
    return coeffs.reshape(1, -1).to(device)

def compute_visuals_rec_only(pred_face, pred_mask, input_img):
    batch_size, channels, height, width = input_img.shape
    black_bg = torch.zeros((batch_size, channels, height, width),
                           device=device, dtype=input_img.dtype)
    output_vis = (pred_face * pred_mask + (1 - pred_mask) * input_img).squeeze(0)
    output_vis_black = (pred_face * pred_mask + (1 - pred_mask) * black_bg).squeeze(0)
    input_np = input_img.squeeze(0).cpu().numpy()
    return output_vis, output_vis_black, input_np

def save_mesh_real(pred_vertex, pred_color, face_buf, name):
    shape = pred_vertex.cpu().numpy()[0]
    shape[..., -1] = 10 - shape[..., -1]
    color = pred_color.cpu().numpy()[0]
    tri = face_buf.cpu().numpy()
    mesh = trimesh.Trimesh(
        vertices=shape,
        faces=tri,
        vertex_colors=np.clip(255.*color, 0, 255).astype(np.uint8)
    )
    mesh.export(name)

vector1 = [-8.50905649e-01, -1.78019962e-01, -2.69281082e-01, -2.18959234e-01, -1.30566012e-01,  2.14945111e-01,
        -2.75066824e-02,  1.08074224e-01, -7.15784775e-03, 1.68816284e-01, -9.14148278e-02,  2.49742995e-01,
        -4.59440145e-01,  2.01890144e-02, -4.57127386e-01, 4.04270663e-01,  4.78660461e-02,  2.99468178e-01,
        -3.79396905e-01, -1.25760746e-01, -2.11134672e-01, -2.01071983e-01, -1.08055768e-02, -8.70057970e-02,
         1.76990089e-01,  1.36777385e-01,  1.17822622e-01, -1.78706219e-01,  1.16125342e-01,  2.52745142e-01,
        -1.50845048e-02,  3.11748943e-01,  1.53720134e-01, 2.38541674e-01, -1.77707052e-01, -9.13623916e-02,
         7.13670607e-02, -5.68063865e-02, -3.27588097e-03, 1.48747174e-01,  3.61475269e-01,  1.16842768e-01,
        -1.21887835e-01,  1.61566299e-01,  1.86472420e-02, -7.77296977e-02,  2.98230936e-02,  6.55554460e-02,
         7.16076604e-02, -3.80762870e-02, -1.99853755e-01, -2.71313102e-02, -1.71870677e-01,  1.42643057e-01,
         9.11511219e-02, -2.51433803e-01, -1.23972300e-01, -9.71749000e-02, -1.32439471e-01,  7.72636153e-02,
        -1.43799765e-01, -3.83887900e-01, -1.28752204e-01, -2.45132466e-01, -1.33727779e-01,  1.08863223e-01,
         5.97718102e-02, -3.32034515e-02, -9.21267759e-02, 1.51556853e-01,  2.19294068e-01, -1.63523977e-01,
         4.87171708e-02, -4.12576267e-02, -2.06937656e-01, 2.28411299e-02, -1.57998831e-01,  3.16625046e-01,
         1.81852016e-01,  4.83241656e-01, -1.21800159e+00, -1.28767409e-01,  1.89552316e+00,  1.84966283e+00,
        -2.35790010e+00, -3.18781783e+00, -1.47634435e+00, 8.49617935e-01, -6.32787584e-01, -5.37433595e-04,
         2.04763748e+00, -1.46675807e-02,  9.98623298e-02, 6.66956828e-01,  1.71931114e+00,  1.95733457e+00,
         6.45365028e-01, -5.30639059e-02,  2.55013566e+00, 4.51026085e-01, -1.47227418e+00,  2.21086301e+00,
         5.23568422e-01,  1.07048234e+00,  1.09852031e+00, 9.88008478e-01, -3.98975374e-02, -1.99631609e+00,
        -1.86991582e-01, -9.14242124e-01,  5.35264407e-01, -1.86993368e+00, -2.16750934e+00,  2.97943416e-01,
        -1.77991088e+00, -7.51779732e-01,  1.34310794e+00, -2.49651421e-01,  1.32790359e-01, -8.92549729e-01,
         1.68269365e+00,  4.30080560e-01, -1.27974086e+00, -9.86305299e-01,  1.03707820e+00, -4.80792188e-01,
         1.19955698e+00,  7.92715918e-01, -1.30173040e+00, 6.21760947e-01, -1.01615686e+00,  7.44691926e-01,
        -9.43839444e-01,  2.99519315e+00,  2.12012789e-01, -2.20101148e-01, -2.26246066e-01,  6.32365578e-01,
         1.03763083e+00, -1.64426593e+00,  6.57543489e-01, -6.90095657e-01, -1.25676628e+00,  1.84143952e+00,
        -2.18667092e+00, -1.02127162e+00, -2.74207707e-01, 5.10690228e-01,  5.51961545e-01,  1.80242913e+00,
         1.61859716e+00, -1.13145993e+00,  3.38822226e+00, 1.32059145e+00,  5.23358263e-02, -8.98875210e-02,
        -2.12715033e+00, -1.64344731e+00, -9.08141714e-01, 2.53417475e-01]

# ########## beauty 4.0 female group2(AF)

vector2 = [-9.96279138e-01, -2.22433215e-01,  5.38392891e-03, 1.18407666e-01,  4.71871691e-01, -1.50604876e-01,
         5.83095933e-03,  1.46658408e-01, -1.02206040e-01, 1.95019037e-01, -7.79932072e-01, -1.11585183e-01,
        -3.25992987e-01, -1.57005051e-01, -8.63374116e-01, 4.18110689e-01,  5.12386694e-02,  2.74293431e-01,
        -5.02146302e-01, -2.24039568e-01, -3.07553656e-01, -1.08735062e-01,  2.12641344e-01,  2.72525296e-02,
         2.58173046e-01,  1.95788759e-01, -2.31352274e-01, -2.81337524e-01, -8.09691474e-02,  1.49753282e-01,
         1.31183435e-01, -2.92438912e-02, -1.79087000e-01, 9.79893715e-02, -7.28434370e-02, -4.44791792e-01,
        -1.67054142e-02,  4.69529674e-01,  1.63694605e-01, 4.07931259e-01,  3.97083949e-01,  1.22316996e-01,
         1.12509990e-01,  2.56759138e-01, -2.40964488e-01, -3.05190727e-02,  2.45124873e-02, -2.91584814e-02,
        -1.10953089e-01,  1.74231678e-01, -2.07321901e-01, 1.19147472e-01,  7.84877927e-02, -3.41322175e-02,
         5.87901592e-01, -1.06415992e-01,  3.19926440e-02, 1.10289850e-01, -1.54878832e-02,  1.17139671e-01,
        -7.37654136e-02, -4.72422643e-02, -2.75967446e-01, -2.26681539e-02, -1.44311376e-01,  6.04320966e-02,
         9.01921419e-03,  2.51063802e-02, -1.72773472e-01, -6.65557915e-03,  2.09553218e-01,  1.32371677e-01,
        -4.88916185e-03, -9.70618098e-02, -2.92383213e-01, 1.50949033e-01,  2.26950330e-01,  5.13660048e-02,
         3.70367058e-01,  4.59426374e-01, -1.22592361e+00, -7.50292480e-01,  1.60237714e+00,  2.24515410e+00,
        -1.74893734e+00, -2.72661737e+00, -5.58271830e-01, 1.26057110e+00, -5.48678357e-01, -1.30044428e-01,
         2.16352109e+00,  4.08983681e-01,  9.15414490e-01, 8.31344261e-01,  1.15545776e+00,  1.90396592e+00,
         8.09756072e-01,  1.88428956e-01,  2.44275501e+00, 6.28677458e-02, -1.93616479e+00,  1.85301888e+00,
         9.67058702e-01,  8.77739872e-01,  3.43674839e-02, 1.74717215e+00, -3.33403170e-01, -2.35561441e+00,
         5.77516462e-02, -8.95872992e-01,  5.82069853e-01, -2.35182991e+00, -1.87474433e+00, -1.25956654e-01,
        -3.12936108e+00, -1.85434464e+00,  1.08851283e+00, -7.38272214e-01, -6.27026690e-02, -1.56372971e+00,
         1.23273932e+00,  6.56056799e-01, -1.39885010e+00, -5.87280671e-01,  6.84856126e-01, -1.34638021e+00,
         1.27427295e+00,  4.48819985e-01, -4.70874143e-01, -1.61462103e-01, -1.14069713e+00,  1.52326878e+00,
        -1.03392469e+00,  2.45677126e+00,  3.95025279e-02, -1.36170382e-01,  9.53382940e-02,  1.10161865e+00,
         5.45449354e-01, -1.59652404e+00,  3.48115590e-01, -8.08700196e-01, -7.15335402e-01,  1.25891252e+00,
        -2.06838278e+00, -5.59051304e-01, -4.49240049e-01, -3.24698047e-01,  2.29098947e-01,  8.41876830e-01,
         2.03008646e+00, -1.19486516e+00,  2.61080486e+00, -2.50869845e-01, -6.24762877e-01, -3.10623060e-01,
        -2.28109023e+00, -1.73361247e+00, -1.55298967e-01, -8.49543537e-01]


# ########## beauty 4.0 male group1(CM)

vector3 = [-7.39056291e-02,  1.99929694e-02, -5.52900977e-01, -6.29202895e-01, -2.64116132e-01,  4.24015643e-01,
         3.99798586e-01, -2.58858672e-01, -9.12366839e-02, -2.26584125e-01, -1.61699861e-01, -4.01062245e-02,
        -4.04006948e-01, -9.24905329e-02,  1.13654550e-01, 2.32771850e-01,  1.10164317e-01,  2.85828028e-01,
         1.04228261e-01, -2.86952014e-01, -2.53193858e-01, -2.44909364e-01, -3.13665022e-02,  1.90624404e-01,
         5.40924220e-01, -8.42221635e-02, -9.41599028e-02, 1.06894137e-01,  1.01487404e-01, -9.03888119e-03,
         8.28707682e-02,  1.97422008e-01, -1.69475921e-01, 5.27025612e-02, -2.32620893e-01, -2.07486139e-01,
        -1.11825435e-01, -3.06263527e-02,  1.96625990e-01, 2.17320405e-01,  3.69377914e-01, -3.00314278e-02,
         7.19684491e-02,  2.37629936e-01, -1.35458732e-01, 5.37051645e-02, -1.56772336e-01,  1.39288158e-02,
        -2.49905179e-01,  1.92005321e-01, -3.99707225e-01, 1.01851979e-04,  1.27706428e-02, -9.52591044e-02,
         5.42079617e-01, -1.24377305e-01, -4.83154273e-02, 1.25708512e-02,  5.76758091e-02,  7.95989813e-02,
         7.42765240e-03, -2.44187976e-01, -3.14596673e-01, 9.42113872e-02, -1.15084823e-04,  2.74355043e-01,
         1.58445355e-02, -6.38810682e-02,  6.56019204e-02, 4.86278209e-02,  3.19519076e-01,  1.26049395e-02,
         7.48541519e-02,  1.84259674e-01, -1.58705390e-01, 2.03322704e-01,  4.20502916e-02,  1.33356054e-01,
         1.86188913e-01,  3.71818060e-01, -3.90571284e-01, -5.00650344e-01,  1.92219314e+00,  1.37742640e+00,
        -2.67238206e+00, -1.77792495e+00, -8.01248327e-01, 1.15750265e-01, -1.02169392e+00, -6.21105250e-01,
         1.55232418e+00,  9.69816329e-01,  5.08648257e-01, 8.21984026e-01,  1.30310967e+00,  2.59369785e+00,
         8.38189907e-01,  2.23109679e-02,  2.69747393e+00, 6.53533473e-02, -1.39166932e+00,  1.83625455e+00,
         1.63701010e+00,  5.24662308e-01,  1.10115831e+00, 1.52025105e+00,  3.67888371e-01, -2.04799122e+00,
         3.12712065e-01, -4.83290144e-01,  1.04233110e+00, -2.03855730e+00, -1.97125391e+00,  5.74054659e-01,
        -1.26472632e+00, -1.57965918e+00,  9.92157956e-01, -1.82902299e+00, -5.73717949e-01, -8.84456761e-01,
         1.67763500e+00,  8.98750033e-02, -8.87151957e-01, -1.01080970e+00,  1.20987719e+00, -6.31840499e-01,
         8.52356044e-01,  6.88096616e-02, -1.87636553e+00, 4.93778821e-01, -5.72439823e-01,  6.30021462e-01,
        -1.32221685e+00,  2.17022275e+00,  2.10740879e-02, -1.65626386e-01, -5.59754643e-01,  2.50753707e-01,
         6.23171859e-01, -1.73132376e+00,  2.41085795e-01, -8.18988876e-01, -1.34528674e+00,  2.05213656e+00,
        -2.55340686e+00, -1.22798162e+00, -5.25925791e-01, 1.33920669e-01,  7.66739624e-01,  9.94988586e-01,
         1.99784047e+00, -1.38760873e+00,  3.58830486e+00, -2.72991866e-01, -5.68057357e-01, -6.59178909e-01,
        -2.19211768e+00, -1.33068754e+00, -8.16975341e-01, -1.09261875e+00]


# ########## beauty 4.0 male group2(AM)

vector4 = [-2.80381642e-01, -9.22056520e-03, -1.73410977e-01, -3.42506836e-01,  2.98624232e-01, -7.80476666e-02,
         5.01700256e-01, -2.13558105e-01, -1.11043330e-01, -5.95539815e-02, -6.40224098e-01, -3.01506620e-01,
        -1.38778054e-01, -1.84322102e-01, -5.00922562e-01, 1.60419747e-01,  9.51041958e-02,  4.12645686e-01,
        -2.05396137e-01, -3.29061889e-01, -3.01527164e-01, -1.78440542e-01,  1.27396771e-01,  1.62879069e-01,
         3.77673370e-01,  3.19254191e-02, -1.68971700e-01, -3.15803819e-01, -9.90945266e-02,  3.96914621e-02,
         2.73274397e-01, -1.12079695e-01, -2.62706144e-01, -2.44858030e-02, -9.78570252e-02, -4.55774426e-01,
        -2.04137146e-01,  3.21483734e-01,  2.08507715e-01, 3.97945538e-01,  3.40286097e-01,  6.87248378e-02,
         3.85998102e-01,  2.83835850e-01, -3.01383208e-01, -5.33569258e-02, -7.30239003e-02,  8.13257842e-02,
        -1.47381503e-01,  2.05791911e-01, -4.21303923e-01, 1.76263642e-01,  1.22166857e-01,  1.47512097e-03,
         8.40999410e-01, -7.52209729e-02,  3.36987425e-02, 9.43994560e-02,  1.09432345e-01,  3.90822515e-02,
         3.86481532e-02, -9.84081074e-02, -3.78780458e-01, 2.88657186e-01, -1.42135461e-01,  2.69228391e-01,
         2.40329000e-02, -9.05027574e-02,  6.47751427e-02, -2.68915292e-02,  2.56122771e-01,  2.32241877e-01,
        -1.20719176e-02,  1.55009892e-01, -8.18711346e-02, 3.52080859e-01,  2.96758412e-01, -1.08821027e-03,
         3.27354920e-01,  4.31932521e-01, -4.72890523e-01, -9.45175961e-01,  1.74682348e+00,  1.71625598e+00,
        -2.05323696e+00, -2.12400621e+00, -5.90221375e-01, 8.32437726e-01, -9.61802588e-01, -2.12166928e-01,
         1.57785674e+00,  1.60868246e+00,  9.96424236e-01, 5.53577508e-01,  1.00786885e+00,  2.45459094e+00,
         1.23521020e+00,  2.30294299e-01,  2.49645143e+00, -5.38139008e-01, -1.62834421e+00,  8.62911115e-01,
         1.68612950e+00,  1.10512442e+00,  5.42078514e-01, 1.83621347e+00, -1.97872015e-01, -2.07862050e+00,
        -5.36693615e-02, -1.62643721e-01,  2.07225728e-01, -2.51497098e+00, -1.45428765e+00, -6.52678552e-01,
        -2.55040971e+00, -1.91183530e+00,  1.11405933e+00, -1.52916257e+00, -4.89337308e-01, -1.11084450e+00,
         1.53654634e+00,  8.31640313e-01, -1.35783655e+00, -8.31232144e-01,  9.44190033e-01, -1.56012055e+00,
         7.96776705e-01,  4.62805029e-01, -1.07633368e+00, 5.53802193e-02, -9.07698454e-01,  1.34302714e+00,
        -1.01283254e+00,  2.61376968e+00, -2.06440669e-01, 2.09630729e-01, -4.57183952e-01,  1.11153281e+00,
         1.16073621e-01, -1.22171592e+00,  4.18986564e-01, -1.13369505e+00, -1.06322580e+00,  1.44370313e+00,
        -2.47256262e+00, -5.04321218e-01, -5.50667489e-01, 6.36828411e-02,  3.48812523e-01,  4.79141577e-01,
         2.37317872e+00, -1.17773755e+00,  2.38348348e+00, -9.63158865e-01, -1.19323132e+00, -8.47740820e-01,
        -1.96917028e+00, -1.48899901e+00, -2.66955999e-01, -1.45437374e+00]

centers = {
    'AF': np.array(vector2),
    'CF': np.array(vector1),
    'AM': np.array(vector4),
    'CM': np.array(vector3),
}

def render_image(save_folder,name,k,vectors,new_vector0, image_path, lm_path,target = "closest" ):
        # 构造保存的图像完整路径
        img_path_black_bg = os.path.join(save_folder, target, str(k), f"{name}.png")
        
        # 如果图像已存在，则跳过后续处理
        # if os.path.exists(img_path_black_bg):
        #     print(f"File {img_path_black_bg} already exists, skipping rendering.")
        #     return
        
        
        
        # # 生成和渲染图像        
        # id_coeffs = new_vector0[:80]
        # exp_coeffs = [0] * 64
        # tex_coeffs = new_vector0[80:]
        # angles = [0] * 3
        # gammas = [0] * 27
        # translations = [0] * 3
        # input_coeff = generate_coeff(id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations)
        
        # 生成渲染--标准姿态
        # id_coeffs = new_vector0[:80]
        # exp_coeffs = [0]*64
        # tex_coeffs = new_vector0[80:]
        # angles = [0]*3
        # gammas = [0]*27
        # translations = [0]*3
        # input_coeff = generate_coeff(id_coeffs, exp_coeffs, tex_coeffs,
        #                              angles, gammas, translations)

        # 生成渲染--原姿态
        id_coeffs = new_vector0[:80]
        exp_coeffs = vectors["exp"]
        tex_coeffs = new_vector0[80:]
        angles = vectors["angle"]
        gammas = vectors["gamma"]
        translations = vectors["trans"]
        input_coeff = generate_coeff(id_coeffs, exp_coeffs, tex_coeffs,
                                     angles, gammas, translations)
        
        pred_vertex, pred_tex, pred_color, pred_lm, pred_lm3dland = facemodel.compute_for_render(input_coeff)
        pred_mask, _, pred_face = renderer(mesh_rasterizer, pred_vertex.float(), facemodel.face_buf.float(), feat=pred_color.float())
        
        input_img, _ = read_data(image_path, lm_path, lm3d_std)
        input_img = input_img.to(device)
        output_img, output_img_black_bg, input_img_raw = compute_visuals_rec_only(pred_face, pred_mask, input_img)
        
        image_numpy_black_bg = tensor2im(output_img_black_bg)
        
        folder_path = os.path.dirname(img_path_black_bg)
 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
 
        save_image(image_numpy_black_bg, img_path_black_bg)
        print(f"save {img_path_black_bg} fishished ")

def load_vectors_from_mat_simple(name, directory):
    file_path = os.path.join(directory, name)
    mat_data = loadmat(file_path)
    return {
        "id":  mat_data["id"].flatten(),
        "tex": mat_data["tex"].flatten()
    }

def move_towards_closest_vector(vector0, centers, k, name=None):
    dists = {n: np.linalg.norm(vector0 - c) for n, c in centers.items()}
    close = min(dists, key=dists.get)
    center = centers[close]
    return vector0 + (center - vector0) * (k / 100)

def move_towards_closest_vector_race(vector0, centers, k, name=None):
    if name and "AF" in name:
        center = centers['AF']
    elif name and "CF" in name:
        center = centers['CF']
    elif name and "AM" in name:
        center = centers['AM']
    elif name and "CM" in name:
        center = centers['CM']
    else:
        dists = {n: np.linalg.norm(vector0 - c) for n, c in centers.items()}
        center = centers[min(dists, key=dists.get)]
    return vector0 + (center - vector0) * (k / 100)

def move_towards_closest_vector_aisa(vector0, centers, k, name=None):
    if name and "F" in name:
        center = centers['AF']
    elif name and "M" in name:
        center = centers['AM']
    else:
        dists = {n: np.linalg.norm(vector0 - c) for n, c in centers.items()}
        center = centers[min(dists, key=dists.get)]
    return vector0 + (center - vector0) * (k / 100)

def move_towards_closest_vector_euro(vector0, centers, k, name=None):
    if name and "F" in name:
        center = centers['CF']
    elif name and "M" in name:
        center = centers['CM']
    else:
        dists = {n: np.linalg.norm(vector0 - c) for n, c in centers.items()}
        center = centers[min(dists, key=dists.get)]
    return vector0 + (center - vector0) * (k / 100)

def process_file(mat_file):
    if not mat_file.endswith('.mat'):
        return
    name = mat_file.split('.')[0]
    vectors = load_vectors_from_mat(mat_file, mat_input_path)
    image_folder = "your_location/rating_face/image"
    detections_folder = "your_location/rating_face/image/detections"
    save_folder = "your_location/id_keep_beauty_improve"
    image_path = os.path.join(image_folder, f"{name}.jpg")
    lm_path    = os.path.join(detections_folder, f"{name}.txt")
    vector0    = np.concatenate([vectors["id"], vectors["tex"]])
    
    # calculate distance for auto
    if "AF" in name:
        center = centers['AF']
    elif "CF" in name:
        center = centers['CF']
    elif "AM" in name:
        center = centers['AM']
    elif "CM" in name:
        center = centers['CM']
    else:
        dists  = {n: np.linalg.norm(vector0 - c) for n, c in centers.items()}
        center = centers[min(dists, key=dists.get)]
    distance = np.linalg.norm(vector0 - center)

    # iterate fixed k values
    # for k in [0, 10, 30, 50, 70, 90]:
    #     new0_closest = move_towards_closest_vector(vector0, centers, k, name)
    #     new0_race    = move_towards_closest_vector_race(vector0, centers, k, name)
    #     new0_aisa    = move_towards_closest_vector_aisa(vector0, centers, k, name)
    #     new0_euro    = move_towards_closest_vector_euro(vector0, centers, k, name)

    #     try:
    #         render_image(save_folder, name, k, vectors, new0_closest, image_path, lm_path, target='closest')
    #     except Exception as e:
    #         print(f"Error (closest) {name}, k={k}: {e}")
    #     try:
    #         render_image(save_folder, name, k, vectors, new0_race,    image_path, lm_path, target='race')
    #     except Exception as e:
    #         print(f"Error (race)    {name}, k={k}: {e}")
    #     try:
    #         render_image(save_folder, name, k, vectors, new0_aisa,    image_path, lm_path, target='asia')
    #     except Exception as e:
    #         print(f"Error (asia)    {name}, k={k}: {e}")
    #     try:
    #         render_image(save_folder, name, k, vectors, new0_euro,    image_path, lm_path, target='euro')
    #     except Exception as e:
    #         print(f"Error (euro)    {name}, k={k}: {e}")

    # auto mode
    row = score_df.loc[score_df['filename'] == f"{name}.jpg", 'score']
    score_val = float(row.values[0]) if not row.empty else 0.0
    k_val     = adaptive_k(score_val, distance, alpha=20)
    k_folder  = 'auto'

    new0_closest = move_towards_closest_vector(vector0, centers, k_val, name)
    new0_race    = move_towards_closest_vector_race(vector0, centers, k_val, name)
    new0_aisa    = move_towards_closest_vector_aisa(vector0, centers, k_val, name)
    new0_euro    = move_towards_closest_vector_euro(vector0, centers, k_val, name)

    try:
        render_image(save_folder, name, k_folder, vectors, new0_closest, image_path, lm_path, target='closest')
    except Exception as e:
        print(f"Error AUTO (closest) {name}: {e}")
    try:
        render_image(save_folder, name, k_folder, vectors, new0_race, image_path, lm_path, target='race')
    except Exception as e:
        print(f"Error AUTO (race)    {name}: {e}")
    try:
        render_image(save_folder, name, k_folder, vectors, new0_aisa, image_path, lm_path, target='asia')
    except Exception as e:
        print(f"Error AUTO (asia)    {name}: {e}")
    try:
        render_image(save_folder, name, k_folder, vectors, new0_euro, image_path, lm_path, target='euro')
    except Exception as e:
        print(f"Error AUTO (euro)    {name}: {e}")

def main():
    mat_files = [f for f in os.listdir(mat_input_path) if f.endswith('.mat')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(process_file, mat_files)

if __name__ == "__main__":
    main()
