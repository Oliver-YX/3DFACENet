## This file has been taken from DECA and modified ##
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import numpy as np
import pandas as pd
from time import time
import argparse
from tqdm import tqdm
import torch
import torchvision
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr,spearmanr
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def calculate_distance(vector0, center):
    """
    Calculate the Euclidean distance between vector0 and the center.

    :param vector0: The original vector.
    :param center: The center vector.
    :return: Euclidean distance.
    """
    return np.linalg.norm(vector0 - center)

import torch

def move_towards_closest_vector_fix(vector0, centers, k, name, device):
    """
    Move vector0 towards the closest center in centers by k percent.

    :param vector0: torch.Tensor 或可转为 Tensor 的 array，shape=[D]
    :param centers: dict[str, Tensor 或 array]，各中心向量
    :param k: float，移动比例（0-100）
    :param name: unused here
    :param device: 设备字符串，比如 'cuda:0' 或 'cpu'
    :return: torch.Tensor，更新后的向量（在同一个 device 上）
    """
    # 1) 转成 Tensor 并搬到同一个设备
    if not torch.is_tensor(vector0):
        vector0 = torch.tensor(vector0, device=device, dtype=torch.float32)
    else:
        vector0 = vector0.to(device, dtype=torch.float32)

    # 2) 把所有 center 也转成 Tensor 并搬到同一设备
    centers_t = {}
    for nm, c in centers.items():
        if not torch.is_tensor(c):
            c_t = torch.tensor(c, device=device, dtype=torch.float32)
        else:
            c_t = c.to(device, dtype=torch.float32)
        centers_t[nm] = c_t

    # 3) 计算到每个 center 的距离（L2 范数）
    distances = {nm: (vector0 - c_t).norm().item()
                 for nm, c_t in centers_t.items()}

    # 4) 找最近的 center
    closest = min(distances, key=distances.get)
    center = centers_t[closest]

    # 5) 沿着方向移动 k%
    new_vector = vector0 + (center - vector0) * (k / 100.0)

    return new_vector


def move_towards_closest_vector(vector0, centers, k, name, device):
    # 确保 vector0 是 torch.Tensor，并移动到同一个设备
    if not isinstance(vector0, torch.Tensor):
        vector0 = torch.from_numpy(vector0)
    
    vector0 = vector0.to(device)

    # 检查 name 是否包含特定的字符串
    if 'AF' in name:
        closest_center = centers['AF']
        print("choose AF center")
    elif 'CF' in name:
        closest_center = centers['CF']
        print("choose to CF center")
    elif 'AM' in name:
        closest_center = centers['AM']
        print("choose to AM center")
    elif 'CM' in name:
        closest_center = centers['CM']
        print("choose to CM center")
    else:
        # 提取中心向量并转换为 torch.Tensor
        center_tensors = torch.stack(list(centers.values())).to(device)

        closest_distance = float('inf')
        closest_center = None
        for center in center_tensors:
            distance = torch.norm(center - vector0)
            if distance < closest_distance:
                closest_distance = distance
                closest_center = center

    closest_center = closest_center.to(device)
    move_distance = (closest_center - vector0) * (k / 100)
    new_vector = vector0 + move_distance
    return new_vector


def move_towards_closest_vector_C(vector0, centers, k, name, device):
    # 确保 vector0 是 torch.Tensor，并移动到同一个设备
    if not isinstance(vector0, torch.Tensor):
        vector0 = torch.from_numpy(vector0)
    
    vector0 = vector0.to(device)

    # 检查 name 是否包含特定的字符串
    if 'F' in name:
        closest_center = centers['CF']
        print("choose to CF center")
    elif 'M' in name:
        closest_center = centers['CM']
        print("choose to CM center")

    closest_center = closest_center.to(device)
    move_distance = (closest_center - vector0) * (k / 100)
    new_vector = vector0 + move_distance
    return new_vector



###100维
######### beauty 4.0 female group1(CF)
vector1 = [ 1.15978542, -0.18746921, -0.54838555,  0.28252789,  0.14911615,
         0.13715184,  0.32338832,  0.59628093, -0.33601529,  0.15627049,
         0.43845711,  0.18972677, -0.13678658, -0.31309351,  0.3483394 ,
         0.20151991,  0.1347867 , -0.12044146,  0.38061408,  0.02475821,
        -0.19801559,  0.04466595, -0.27473035,  0.35638952,  0.21571366,
         0.21721786, -0.37273863, -0.26375808,  0.10644071,  0.58805257,
        -0.58165938, -0.11223349, -0.30361627,  0.03666466, -0.27723517,
        -0.5126322 ,  0.49285817, -0.15670349, -0.07075035, -0.50564147,
         0.16176677,  0.22067161, -0.28506672, -0.10061107,  0.19119324,
        -0.08846669,  0.21151715,  0.37155274, -0.27878156, -1.02224115,
         0.89748489,  0.91923807,  0.17767841,  0.09853138,  0.35058293,
        -0.7163693 ,  0.80681534, -0.43104842,  0.14543533,  0.61418263,
        -0.5308427 , -0.40134584,  0.37971615, -0.06638459,  0.35050952,
         0.14895452, -0.76183684,  0.62977417,  0.18820314, -0.21208338,
         0.05202342, -1.31923083, -0.95494844,  0.23951194,  0.28517389,
        -0.28436655,  0.48740404, -0.47458455,  0.2584208 ,  0.24636292,
         0.03410638,  0.50421342,  0.06451041,  0.96734548, -0.77137474,
        -0.65261046,  1.08625445, -1.17665791,  0.36318944, -0.28348342,
        -0.89093598,  0.37262   , -0.35898325,  0.38682841, -0.58011712,
        -0.11255396, -0.1996014 , -0.05488406,  0.1911533 ,  0.60878193,
         0.04998604, -0.5767769 , -0.50897023, -0.10032622, -0.05966392,
         0.19837729, -0.06567782,  0.43718012, -0.61019416,  0.34131882,
        -0.16645339, -0.81053725,  0.18561835, -0.36490908,  0.22460781,
         0.13175067,  1.00653449, -0.41156333,  0.65071314,  0.41050743,
         0.18769865,  0.50507042, -0.1474142 , -0.24500221, -0.01819379,
        -0.98406905,  0.25909605, -0.52492792,  0.62334271, -0.29618652,
         0.25424453,  0.2976272 , -0.66860538, -0.08556026,  0.16225039,
         0.15839601, -0.06988789,  0.57208383,  0.75470126,  0.47628464,
        -0.59848432, -0.99614418, -0.14744353,  0.09304517, -0.45427522,
         0.15439181, -0.70794695,  0.19735833,  0.46907437,  0.1036228 ,
         0.50450955,  0.40251444, -0.32719643, -0.51318632, -0.20988478,
         0.15847969,  0.22775173,  0.88847114,  0.01659172, -0.49536991,
        -0.24551229, -0.2705173 ,  0.26963984, -0.40017016, -0.02483421,
         0.89764455, -0.06375391,  0.19752094, -0.17788452,  0.46467521,
        -0.10165918, -0.26715501, -0.40714778, -0.58875508,  0.42317752,
         0.06615507,  0.08759273,  0.22894606, -0.29950354, -0.63834658,
         0.2683623 ,  0.26932846, -0.765364  ,  0.13087018, -0.54912878,
         0.24255656, -0.43584918,  0.45867385,  0.24133871, -0.0029094 ,
         0.52005969,  0.1740307 , -0.15763962, -0.21928988,  0.38875314,
         0.39835836,  0.81434147, -0.30344682, -0.32269973,  0.01754513,
         0.43915186,  0.21726727,  0.1743946 ,  0.90043398,  0.01947816,
         0.13305536, -0.02006686,  0.25716993, -0.04585975,  0.01055574,
         0.26952632,  0.42699803, -0.81691078,  0.31504586,  0.78382515,
         0.29066278, -0.17414934,  0.27581162,  0.0797811 , -0.46669199,
        -0.27583932, -0.13997609, -0.55139683, -0.39838277,  0.16214673,
        -0.18805805,  0.05800843,  0.71162224,  0.38877654, -0.19228815,
         0.01655759, -0.26315817,  0.13651977,  0.65137797,  0.8583561 ,
        -0.43915072,  0.0523555 ,  0.12156365, -0.34109392,  0.25558759,
        -0.37637722, -0.15381801, -0.48872813, -0.32464992, -0.2977804 ,
         0.3561352 ,  0.24614061, -0.48909671, -0.09840634, -0.11545545,
         0.19807881, -0.47562714,  0.3873749 , -0.30385317,  0.41816369,
        -0.38131772,  0.36577996, -0.05331593,  0.34294132, -0.24579097,
        -0.09041336,  0.70994488,  0.47696891,  0.72809791,  0.0614053 ,
        -0.27511953, -0.06623809,  0.1308686 ,  0.29545781,  0.36793119,
         0.13053387, -0.44961808,  0.21767536, -0.66122076,  0.52140292,
        -0.24364373, -0.72450612, -0.93279923, -0.23210801, -0.50106381,
        -0.46002392, -0.66779723,  0.10688708,  0.70872861,  0.33379556,
        -0.60028372,  0.23190072,  0.74211621, -0.0452338 , -0.34684101,
         0.06990861,  0.35135964,  0.37298067,  0.32011946,  0.59215158,
         0.14607644,  0.19628325, -0.1892097 ,  0.2010224 ,  0.26366361]

# ########## beauty 4.0 female group2(AF)af70
vector2 = [ 0.8823727 ,  0.31105343,  0.05601344, -0.03918447,  1.30501111,
         0.35941842, -0.76510013,  0.64114987, -0.49140049,  0.56952761,
         0.82613687, -0.67174215, -1.22405188, -0.8789591 ,  1.24578108,
         0.3238274 ,  0.8925086 , -0.12531974,  0.73674939, -1.12325026,
         0.40194534, -0.74911132,  0.40324331,  1.00757244,  0.63736351,
         0.26841589, -0.70874892,  0.03810627,  0.39212891,  0.30082904,
        -0.86145961, -0.11369903,  0.15837344,  0.14041575, -0.25342039,
        -0.72073177,  0.22119918, -0.84937935, -0.04517507, -0.30806723,
         0.03391626,  0.58848959, -0.53458025, -0.27868613,  0.25433217,
        -0.26365774, -0.34106761,  0.19904682, -0.53341788, -0.85915207,
         1.2573254 ,  0.11726185,  1.0397229 , -0.22733443,  0.92175433,
        -0.34199526,  0.64390607, -0.71955907, -0.04857878,  1.05427346,
        -0.37224029, -1.45562998, -0.02091181, -0.18930595,  1.16030746,
        -0.03475848, -0.14303347,  0.1935044 , -0.66429883, -0.3534979 ,
         0.0018323 , -1.34642563, -0.97417849,  0.45979169,  0.29966867,
         0.18439249,  0.23031693, -0.83933441, -0.20286613,  0.61423252,
         0.0532703 ,  0.52643397,  0.30664454,  1.03144137, -0.73949392,
        -0.43157076,  1.07047423, -1.0410661 ,  0.6619487 , -0.23718991,
        -1.03417424,  0.58560548, -0.18002322, -0.09925426, -0.29212215,
        -0.50114044,  0.57950078,  0.11427733,  0.38793496,  0.68049858,
         0.58091983, -0.834637  , -0.41990125,  0.04772825,  0.00799628,
        -0.21241384, -0.32431758, -0.10717651, -0.82919737,  0.28636076,
        -0.41483155, -1.23860671,  0.18747909,  0.42655239,  0.3365206 ,
         0.19883045,  0.81550335, -0.03081329, -0.19846818,  0.42353698,
         0.42354839,  0.95922817,  0.18855281, -0.33697159,  0.04748654,
        -0.94254716,  0.41049318, -0.49978959,  0.56200617, -0.2234545 ,
         0.60273891,  0.50552074,  0.35289711,  0.23400746, -0.14120669,
         0.68246345, -0.31953764,  0.71646612,  0.55015913,  0.44585274,
        -0.78787922, -0.80840334,  0.17202321,  0.43669991, -0.58088421,
         0.16751565, -0.27468101, -0.07538977,  0.49383017,  0.21949422,
         0.57198495,  0.58107125, -0.05508261, -0.19181209, -0.40888174,
        -0.0467433 ,  0.21178893,  0.81101895, -0.07434704, -0.57132877,
        -0.16600013, -0.68709134, -0.08111172, -0.53833826, -0.41816994,
         0.74068348,  0.14633303, -0.35787224, -0.20761085,  0.1176602 ,
        -0.04201383, -0.08602007, -0.29995518, -0.26065043,  0.20423571,
         0.508366  ,  0.6350786 ,  0.15938314, -0.71427467, -0.63829911,
         0.6947504 ,  0.40132565, -0.07394327, -0.37897604, -0.12190227,
        -0.01203759,  0.33854133,  0.64007357,  0.45150565,  0.19873532,
        -0.03496581,  0.41125005, -0.37376304, -0.66584683,  0.07055023,
         0.31346707,  0.74576002, -0.24584534, -0.38267464, -0.19309984,
        -0.01304782,  0.28114587, -0.30748971,  0.87874472,  0.18274905,
         0.26078157,  0.16029498,  0.51569936, -0.25537834, -0.46948136,
        -0.1395802 ,  0.5970101 , -0.77761301,  0.46688726,  1.02743718,
         0.14627255, -0.34231978,  0.14214374,  0.08565049, -0.28992849,
        -0.1008871 , -0.39843531, -0.26635324, -0.67223578, -0.19707137,
         0.02889581,  0.69416332,  0.14750526,  0.74154688,  0.06329661,
        -0.26689008,  0.1832368 ,  0.62222907,  0.56425575,  0.62893845,
        -0.83424659, -0.5571684 , -0.3385245 , -1.31002739,  0.15642239,
        -0.38945728, -0.2565834 , -0.44642864, -0.7633388 ,  0.019814  ,
        -0.06390205,  0.42705525, -0.5375733 ,  0.13104925,  0.41363252,
         1.00845002, -0.20679099,  0.16395038,  0.27212893,  0.4566363 ,
        -0.42131895,  0.39735234, -0.31841395,  0.14493259, -0.38132162,
        -0.06756699,  1.46017579,  0.0844471 ,  1.30245112,  0.4323942 ,
         0.21414498, -0.49350666,  0.44804452,  0.28701164,  0.63838166,
         0.43627691,  0.06644755,  0.79844494, -0.75122675,  0.93364153,
        -0.09091629, -0.47590651, -0.34414093, -0.22156599, -0.23684022,
        -0.75606838, -1.340315  , -0.09538359,  0.67395026,  0.94200584,
        -0.39124822,  0.33691173,  0.97040167,  0.06541977, -0.34861234,
         0.16769661,  0.30409894, -0.48557984, -0.1129529 ,  0.39380174,
         0.24611555, -0.01144387, -0.00451627,  0.67794823,  0.56548913]



# ########## beauty 4.0 male group1(CM)
vector3 = [-1.84814199e-01, -4.64322709e-01, -8.13722069e-01,
         4.86374453e-01, -4.75880045e-01,  5.13830285e-01,
         3.60504042e-01, -9.49968865e-02, -2.80577095e-01,
         4.10833466e-01,  7.40373339e-01,  5.42304854e-02,
         1.51439132e-02, -2.87461778e-02, -8.19259802e-02,
         2.37340792e-01, -1.89864686e-01, -4.81467301e-01,
         3.39085624e-01,  4.84781895e-01, -3.37854762e-01,
         1.02036501e-01, -2.63635454e-01,  7.00837141e-02,
         2.83176606e-01,  5.17409527e-01, -4.30453824e-01,
        -2.94752061e-01,  1.86237647e-01,  8.33924441e-01,
         2.36867672e-01,  6.63206717e-02, -4.26872886e-01,
        -4.42382456e-01, -6.12144308e-01, -5.88149280e-01,
         1.15651798e-01, -2.37156284e-01, -2.60143815e-01,
        -8.22007503e-01,  1.01590474e-01,  4.39965742e-01,
        -3.98546769e-01, -5.72176967e-01,  9.00743988e-01,
        -1.54710076e-01,  3.68595122e-01,  7.13759483e-01,
        -4.35356604e-01, -9.33956844e-01,  1.03360226e+00,
         8.52749458e-01,  2.73472938e-02, -5.31910052e-02,
         4.81260575e-01, -3.72339554e-01,  5.93709538e-01,
        -2.44192092e-01, -5.23989545e-03,  4.75421555e-01,
        -2.29549185e-01, -5.93396791e-01,  1.86112009e-01,
        -4.81232341e-01,  8.48229364e-02,  2.75683491e-02,
        -7.33188678e-01,  4.74395914e-01,  2.98508445e-01,
         2.14314186e-01, -3.93400649e-01, -1.44508147e+00,
        -9.99216519e-01,  3.81547690e-01, -1.07130573e-01,
        -7.25699093e-01, -5.16059902e-02, -1.59831104e-01,
         4.10062130e-01,  1.14633596e-01,  3.30392999e-01,
         3.38932222e-01, -3.46807994e-02,  9.40486242e-01,
        -9.25837652e-01, -3.30167526e-01,  7.51190142e-01,
        -1.06862308e+00, -7.90083200e-02, -2.94864599e-01,
        -5.25291250e-01,  1.88372852e-01,  1.96793882e-01,
         1.40028333e-02, -4.75709260e-01, -2.84018323e-01,
        -8.56635129e-04,  1.82278044e-01,  5.56284488e-01,
         6.67228616e-01, -2.03924887e-01, -1.37728594e-01,
        -2.16448484e-01, -9.60327819e-02,  6.26028833e-02,
         1.66428419e-01, -2.11610577e-01,  6.08659299e-01,
        -4.76226634e-01,  4.03626971e-01, -6.43914940e-01,
        -5.53967137e-01, -8.83322929e-02, -9.54643118e-02,
         7.27619063e-02, -5.24842083e-01,  6.52762152e-01,
        -5.49680541e-01,  2.39175585e-01,  5.13108300e-01,
         1.34762541e-02,  2.08802248e-01, -1.01302528e-01,
        -4.82940980e-01,  3.66995693e-02, -1.07443057e+00,
         2.37859148e-01,  2.65258931e-01,  5.11142622e-01,
        -1.16723828e-01,  2.72284020e-01,  1.45537021e-01,
        -5.43455601e-01,  5.00782507e-01, -1.50419245e-01,
         5.16654430e-01, -5.28155667e-02,  4.41808969e-01,
         2.26625718e-01,  1.02145810e-01, -6.16606578e-01,
        -8.68061312e-01, -1.45815314e-01,  2.53204329e-02,
         1.75968111e-01,  1.76761931e-01, -7.91066741e-01,
         1.61939233e-01, -1.09704712e-01,  1.56226482e-01,
         2.82160630e-01,  5.20176266e-01, -6.03778907e-01,
        -6.74212954e-01,  3.76028465e-02,  4.38423483e-01,
        -1.03011488e-01,  4.79017370e-01, -1.19637505e-01,
        -4.90110149e-01, -2.19330825e-01, -9.90267755e-02,
         4.51450254e-01, -6.96846240e-01, -4.70385206e-01,
         5.69543679e-01, -2.04072192e-01, -6.72194060e-02,
        -1.13222663e-01, -1.42406777e-02, -1.64891004e-01,
        -2.96341687e-01, -3.18122582e-01, -8.68425400e-01,
         5.09046973e-01,  2.70961035e-01, -3.31282817e-03,
         1.03550518e-01, -2.56850791e-02, -2.60830509e-01,
         2.85218371e-01, -2.15018515e-01, -6.88076334e-01,
        -2.19584182e-01, -1.78767813e-01, -1.89164730e-01,
        -2.98408181e-01,  1.45288042e-01,  3.09855761e-01,
        -4.27152285e-01,  2.19960112e-01,  1.66529298e-01,
        -1.32464202e-01,  4.23689706e-01,  4.81808366e-01,
        -1.68110846e-02,  7.17038638e-01, -2.53621294e-01,
        -1.24045962e-01, -8.75673772e-02,  4.32164701e-01,
         2.70638655e-01,  3.66870231e-01,  5.56644286e-01,
        -1.90634065e-01, -2.28226403e-01,  4.39894221e-02,
        -5.83053444e-02,  5.74575118e-02,  2.87447517e-01,
         9.37291098e-03,  2.19203358e-01, -8.04959058e-01,
         5.61147947e-01,  7.72882137e-01,  6.59262322e-01,
        -1.96433120e-01, -4.48335203e-02,  3.96097919e-01,
        -5.38081418e-01, -9.21700099e-02, -1.43011699e-01,
        -4.84814025e-01, -3.08498968e-01, -7.66310473e-02,
        -2.19325588e-01,  2.58711717e-01,  5.33606775e-01,
         4.11811414e-01, -1.63406447e-01, -2.27014636e-01,
        -2.80472678e-01, -5.98065062e-02,  2.73561003e-01,
         6.95114709e-01, -1.97708041e-01,  4.69314854e-01,
        -3.39156471e-02, -1.66756284e-01,  1.28522659e-01,
        -5.00312573e-01, -4.91523604e-01, -4.51991940e-01,
        -1.80485406e-01, -6.47908418e-01,  7.59733685e-02,
         6.35632597e-01, -1.52236154e-01, -1.61793553e-01,
        -1.21744789e-01,  3.18108664e-01, -4.90394620e-01,
         8.79805707e-01, -6.55620962e-01,  2.81542679e-01,
        -6.42955813e-01, -1.83763024e-01,  3.23754846e-01,
         1.45864638e-01,  1.79225034e-01, -2.60619995e-01,
         2.39540828e-01,  3.56336359e-01,  4.53748264e-01,
         3.85195262e-01,  7.10047653e-04,  2.28756183e-01,
         2.25221019e-01,  5.04655000e-01,  2.86943909e-01,
         3.91345418e-01, -7.91293301e-01,  8.11714010e-02,
        -3.80689376e-01,  4.64011540e-01, -5.95379064e-03,
        -5.02336922e-01, -1.01348507e+00, -3.87065323e-01,
        -8.11533709e-01, -8.21682041e-01, -2.09548975e-01,
         2.74024263e-02,  5.99671794e-01, -1.74826681e-03,
        -5.93073547e-01,  5.95082554e-02,  1.23528510e-01,
         8.79719938e-02, -2.71440426e-01, -9.66606343e-02,
         1.26946413e-01,  6.20659009e-01,  3.40276565e-01,
         7.37821033e-01,  3.88771669e-01,  2.36985901e-02,
        -1.63903630e-02,  4.92441785e-02,  3.41959033e-01]


# ########## beauty 4.0 male group2(AM)am117
vector4 = [-8.44045276e-02, -1.72297525e-01, -1.98992813e-01,
         2.84452505e-01,  7.47487858e-01,  7.64804407e-01,
        -7.26983998e-01,  5.66024173e-02, -4.46412538e-01,
         9.59954118e-01,  9.68394535e-01, -6.73368570e-01,
        -1.11808837e+00, -6.09283995e-01,  1.06484661e+00,
         4.00111153e-01,  7.10221925e-01, -2.61943928e-01,
         9.65274142e-01, -1.05091508e+00,  1.31800528e-01,
        -5.95918717e-01,  1.95897942e-01,  8.72303388e-01,
         1.81933153e-01,  4.96904012e-01, -5.39317697e-01,
         9.91932606e-02,  4.38420703e-01,  2.01901807e-01,
        -1.39880337e-01,  1.38952055e-01, -4.53759250e-02,
        -1.15107024e-01, -4.27713276e-01, -7.11114453e-01,
        -2.43744270e-01, -7.85602790e-01, -9.78268851e-02,
        -3.64718958e-01, -8.70835940e-02,  8.82504226e-01,
        -5.78442870e-01, -6.74734814e-01,  9.94144224e-01,
        -2.98990897e-01, -3.46967622e-01,  4.86829316e-01,
        -9.09822511e-01, -8.31865259e-01,  1.62553599e+00,
        -2.81331630e-01,  8.72840326e-01, -4.85534400e-01,
         1.00783983e+00, -1.98699653e-01,  3.76960809e-01,
        -7.29974351e-01, -1.79196090e-01,  9.18987666e-01,
        -3.34532960e-01, -1.63716196e+00, -1.02480098e-01,
        -5.00153412e-01,  9.42534623e-01, -1.19454021e-02,
        -2.45832229e-01,  1.66825257e-01, -4.87898363e-01,
        -6.44330641e-02, -4.86194738e-01, -1.55338085e+00,
        -9.08929058e-01,  5.36946785e-01, -1.50400406e-02,
        -3.04777769e-01, -3.34104835e-01, -5.96613447e-01,
         1.01124566e-01,  3.79081166e-01,  4.49301071e-01,
         5.24933259e-01,  2.52031549e-01,  9.97589495e-01,
        -7.72981992e-01, -1.16169077e-01,  5.95205505e-01,
        -1.03640190e+00,  3.70005309e-01, -3.73716110e-01,
        -6.81445421e-01,  1.41289405e-01,  3.33272989e-01,
        -4.06554891e-01, -2.50144115e-01, -5.49633111e-01,
         6.17985163e-01,  4.06212393e-01,  7.55401919e-01,
         8.43927517e-01,  2.61635076e-01, -5.43259215e-01,
        -2.01103297e-01,  1.92849162e-01,  2.17548422e-02,
        -2.21991266e-01, -4.99251116e-01,  8.96747963e-02,
        -5.99599495e-01,  5.98118428e-01, -7.14827540e-01,
        -9.69226899e-01,  1.13233061e-02,  5.65032312e-01,
         3.84271223e-02, -3.85792528e-01,  5.90722892e-01,
        -2.15723555e-01, -3.52492737e-01,  3.22521472e-01,
         2.90715398e-01,  7.12431532e-01,  1.08984619e-01,
        -4.59629588e-01,  1.18576261e-01, -9.88303124e-01,
         2.51291712e-01,  2.72584023e-01,  5.23205572e-01,
        -5.91615786e-02,  6.25177142e-01,  4.95542003e-01,
         2.46412446e-01,  6.54674426e-01, -3.43967251e-01,
         9.06270712e-01, -2.57284232e-01,  4.81099568e-01,
         8.22713213e-02,  1.08461889e-01, -8.52676243e-01,
        -5.72370758e-01,  9.35124338e-03,  1.21766584e-01,
        -4.79023747e-02,  1.62856343e-01, -4.62818719e-01,
        -9.02550175e-02, -9.31680285e-02,  2.22626089e-01,
         4.15141149e-01,  6.53666447e-01, -2.28672400e-01,
        -3.91596461e-01, -1.41557381e-01,  1.79290943e-01,
        -1.40759258e-01,  3.77474791e-01, -7.23452144e-02,
        -6.77532698e-01, -1.03670452e-01, -5.38998193e-01,
         7.24890195e-02, -8.03188121e-01, -6.30179479e-01,
         4.10225127e-01,  4.84155480e-02, -5.28766342e-01,
        -1.16417398e-01, -2.83436570e-01, -2.07748351e-01,
        -6.69975838e-02, -2.80916892e-01, -4.35786537e-01,
         2.70869390e-01,  5.97766716e-01,  4.71727201e-01,
        -3.41620476e-02, -5.49521647e-01, -3.56722098e-01,
         6.32551606e-01, -1.18233365e-01, -3.57390700e-02,
        -7.08459986e-01,  2.97488936e-01, -4.06544350e-01,
         5.35996964e-01,  3.35957313e-01,  2.76114778e-01,
        -2.90198854e-01, -3.35516723e-01,  4.66957968e-01,
        -2.56830474e-01, -7.68450579e-03,  3.22947374e-01,
        -5.06850240e-02,  7.19964341e-01, -3.69452817e-01,
        -2.09276850e-01, -3.56678426e-01, -4.91332207e-02,
         3.69324736e-01, -2.05854289e-01,  5.83690453e-01,
        -1.74931118e-01, -9.59497968e-02,  2.58512768e-01,
         4.13338291e-01, -5.40892661e-02, -3.89923907e-02,
        -3.70211927e-01,  4.33804881e-01, -6.98741160e-01,
         6.99028252e-01,  8.53002343e-01,  4.88472130e-01,
        -3.03734157e-01, -1.65255133e-01,  4.69363495e-01,
        -1.61715328e-01, -1.04533003e-01, -2.81165964e-01,
        -1.72034794e-01, -7.43859670e-01, -1.48705815e-01,
        -1.64889825e-01,  7.22685383e-01, -3.98221804e-02,
         7.02421002e-01,  2.47855104e-01, -4.99293338e-01,
         3.42465944e-02,  4.62998757e-01,  1.69350638e-01,
         3.97645357e-01, -6.50265151e-01, -1.52118484e-01,
        -5.35206333e-01, -1.21669916e+00,  1.22743600e-01,
        -4.68214756e-01, -4.59655753e-01, -2.24445767e-01,
        -5.90339656e-01, -2.55320933e-01, -3.18855372e-01,
         7.14194649e-01, -6.27905851e-02,  1.45397080e-01,
         2.14815025e-01,  1.09937085e+00, -3.37245927e-01,
         5.97162502e-01, -6.80515663e-03,  3.44766437e-01,
        -9.10073970e-01,  1.24146185e-01, -2.59941310e-02,
         4.56134067e-02, -1.11162654e-01, -1.45890491e-01,
         1.02518760e+00,  2.11537944e-01,  1.00682634e+00,
         8.38662820e-01,  3.83033600e-01, -2.41841720e-02,
         5.49637840e-01,  5.14142504e-01,  6.91657093e-01,
         6.71098724e-01, -3.83856507e-01,  6.52670396e-01,
        -7.23412111e-01,  8.11716902e-01, -1.00888939e-03,
        -2.03403652e-01, -2.83645872e-01, -3.55482926e-01,
        -5.50592461e-01, -7.94988555e-01, -8.68045032e-01,
        -2.18745280e-01,  5.53418791e-01,  8.23015298e-01,
        -3.76079636e-01,  1.22038983e-01,  2.52233263e-01,
         1.08174882e-01, -3.80263029e-01,  3.45611587e-02,
         2.42815736e-01, -2.49482112e-01,  3.26495233e-02,
         3.23589321e-01,  3.20051772e-01, -3.38999309e-01,
         8.00868058e-02,  5.02039707e-01,  5.86515033e-01]

##### 聚类中心
centers = {
    'AF': torch.from_numpy(np.array(vector2)).float(),
    'CF': torch.from_numpy(np.array(vector1)).float(),
    'AM': torch.from_numpy(np.array(vector4)).float(),
    'CM': torch.from_numpy(np.array(vector3)).float()
}
 
# ################################################         美化示例           ####################################
######  

def deca_generate(args):

    savefolder = args.savefolder
    print(savefolder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    os.makedirs(savefolder, exist_ok=True)
    args.use_mica = True
    
    ####single
    #AM58
    # name = args.name
    # mat_name = f"{name}.mat"
    # img_path = os.path.join(args.inputpath, f"{name}.png")
    
    ####multy
    img_path = args.inputpath
    k = args.k  # 移动 50% 的距离
    
    testdata = datasets.TestData(img_path, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step, crop_size=1024,
                                 use_mica=args.use_mica)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device, use_mica=args.use_mica)
    
    args.saveDepth = False
    args.saveKpt = False
    args.saveObj = True
    args.saveMat = False             #    True
    args.saveVis =  False # True             #   
    args.saveImages = True
    args.render_orig = True
    
    for i in tqdm(range(len(testdata))):

        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]
        arcface_inp = testdata[i].get('arcface_inp', None)

        with torch.no_grad():

            if arcface_inp is not None:
                arcface_inp = arcface_inp.to(device)[None, ...]

            codedict = deca.encode(torchvision.transforms.Resize(224)(images), arcface_inp)
            codedict['images'] = images
        
        ##### 还需要修改deca.py对应代码
        # acc = "fine" 
        acc = "croase" #"fine"
        
        # 种族对应
        codedict["mica_shape"] = move_towards_closest_vector_fix(codedict["mica_shape"], centers, k, name, device)
        ## 同一欧洲
        # codedict["mica_shape"] = move_towards_closest_vector_C(codedict["mica_shape"], centers, k, name, device)
        codedict["mica_exp"] = torch.zeros_like(codedict["mica_exp"]).to(device)
        
        modify_vector_path = os.path.join(savefolder, name, str(k), acc , f'modify_{name}_{k}.txt')
        
        formatted_vector = str(list(codedict["mica_shape"].cpu().numpy()))
            
        # os.makedirs(os.path.dirname(modify_vector_path), exist_ok=True)

        # with open(modify_vector_path, 'w') as file:
        #     file.write(formatted_vector)

        # for key in codedict.keys():
        #     codedict[key] = codedict[key].float().to(device)
            
        opdict, visdict = deca.decode(codedict, name, render_orig = args.render_orig, original_image = codedict['images'] ) #tensor 
        

        # 处理 opdict 中的张量，确保它们是 tensor 类型并且在同一设备上
        for key in opdict.keys():
            if isinstance(opdict[key], torch.Tensor):
                opdict[key] = opdict[key].detach().cpu()
        
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))

        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())

        if args.saveObj:
            objpath = os.path.join(savefolder,str(k), name, name + f'_{k}.obj')
            os.makedirs(os.path.dirname(objpath), exist_ok=True)
            ## save croase
            deca.save_obj(objpath, opdict, codedict)
            # ## save detail
            # deca.save_obj(objpath, opdict, codedict)


        if args.saveMat:
            # opdict = util.dict_tensor2npy(opdict)
            # savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
            codedict_numpy = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in codedict.items()}
            # savemat(os.path.join(savefolder, name, name + '.mat'), codedict_numpy)
            ## save croase
            savemat(os.path.join(savefolder, acc, 'mat', name + '.mat'), codedict_numpy)
            # ## save detail
            # savemat(os.path.join(savefolder,'detail', 'mat', name + '.mat'), codedict_numpy)
            
        if args.saveVis:
            vispath = os.path.join(savefolder, 'vis', str(k), name + f'_{k}_vis.jpg')
            os.makedirs(os.path.dirname(vispath), exist_ok=True)
            # correct_color = cv2.cvtColor(deca.visualize(visdict)) 
            # cv2.imwrite(os.path.join(savefolder, name, name + '_vis.jpg'), correct_color)
            # cv2.imwrite(os.path.join(savefolder, name, name + '_vis.jpg'), deca.visualize(visdict))
            # ## save croase
            # cv2.imwrite(os.path.join(savefolder, 'croase' ,'vis', name + '_vis.jpg'), deca.visualize(visdict))
            ## save detail
            cv2.imwrite(vispath, deca.visualize(visdict))
            

            # for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images']:
                if vis_name not in visdict:
                    continue
                # 确保输出目录存在
                vis_dir = os.path.join(savefolder, str(k), 'vis', vis_name)
                os.makedirs(vis_dir, exist_ok=True)

                # 构建文件路径并保存
                imgpath = os.path.join(vis_dir, f"{name}.jpg")
                img = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(imgpath, img)

            
    print(f'-- please check the results in {savefolder}')



        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlbedoGAN')

    parser.add_argument('-i', '--inputpath', default='inference_test/in_data', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='inference_test/out_data', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    parser.add_argument('--use_mica', default=False, action="store_true", help="whether to use ArcFace backbone for inference")
    
    parser.add_argument('--name', default='None', type=str, help="input image to beautify")
    
    parser.add_argument('--k', default='0',  type=int, help="the level of beautification")
    
    args = parser.parse_args()

    deca_generate(args)