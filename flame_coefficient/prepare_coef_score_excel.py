import os
import scipy.io
import numpy as np
import pandas as pd

save_path = "/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_coefficient"
if not os.path.exists(save_path):
    os.makedirs(save_path)

def load_data(file_path, directory):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            file_name, score = line.strip().split()
            mat_file = os.path.join(directory, file_name.replace('.jpg', '.mat'))
            if os.path.exists(mat_file):
                mat_data = scipy.io.loadmat(mat_file)
                row = [file_name, float(score)]
                row.extend(mat_data["mica_shape"].flatten())
                # row.extend(mat_data["tex"].flatten())
                data.append(row)
    return data

def merge_excel(file1, file2):
    # 读取两个 Excel 文件
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # 合并两个 DataFrame
    merged_df = pd.concat([df1, df2], ignore_index=True)
    return merged_df

def save_to_excel(data, file_name):
    columns = ['filename', 'score']
    columns.extend([f'mica_shape_{i}' for i in range(1, 301)])
    # columns.extend([f'tex_{i}' for i in range(1, 81)])
    df = pd.DataFrame(data, columns=columns)
    df.to_excel(os.path.join(save_path, file_name), index=False)

# 路径设置
# directory = '/workspace/AdBRC/xieyuan/3Dtask/Deep3DFaceRecon_pytorch/checkpoints/cvpr_v11/results/image/epoch_75_000000/'
directory = '/workspace/AdBRC/xieyuan/data/5500_AlbedoGAN/croase/mat'
train_file = '/workspace/AdBRC/xieyuan/3Dtask/SCUT-FBP5500-Database-Release/train_test_files/split_of_60%training and 40%testing/train.txt'
test_file = '/workspace/AdBRC/xieyuan/3Dtask/SCUT-FBP5500-Database-Release/train_test_files/split_of_60%training and 40%testing/test.txt'

# 加载数据
print("Loading training data...")
train_data = load_data(train_file, directory)
print("Loading testing data...")
test_data = load_data(test_file, directory)

# 保存到 Excel
save_to_excel(train_data, 'train_coef.xlsx')
save_to_excel(test_data, 'test_coef.xlsx')

merge_data = merge_excel("/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_coefficient/test_coef.xlsx","/workspace/AdBRC/xieyuan/Beautytask/3DFACENet(IJCAI)/flame_coefficient/train_coef.xlsx")
save_to_excel(merge_data,'all_coef.xlsx')