import os
import scipy.io
import numpy as np
import pandas as pd

save_path = "your_save_path_for_coefficient"
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
                row.extend(mat_data["id"].flatten())
                row.extend(mat_data["tex"].flatten())
                data.append(row)
    return data

def merge_excel(file1, file2):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    merged_df = pd.concat([df1, df2], ignore_index=True)
    return merged_df

def save_to_excel(data, file_name):
    columns = ['filename', 'score']
    columns.extend([f'id_{i}' for i in range(1, 81)])
    columns.extend([f'tex_{i}' for i in range(1, 81)])
    df = pd.DataFrame(data, columns=columns)
    df.to_excel(os.path.join(save_path, file_name), index=False)

# path setting
directory = 'save_path_for_3D_facial_reconstruction_result'
train_file = 'SCUT-FBP5500-Database-Release/train_test_files/split_of_60%training and 40%testing/train.txt'
test_file = 'SCUT-FBP5500-Database-Release/train_test_files/split_of_60%training and 40%testing/test.txt'

# load data
print("Loading training data...")
train_data = load_data(train_file, directory)
print("Loading testing data...")
test_data = load_data(test_file, directory)

# save to Excel
save_to_excel(train_data, 'train_coef.xlsx')
save_to_excel(test_data, 'test_coef.xlsx')

# Merge train/test coefficient Excel files (paths under save_path)
train_coef_file = os.path.join(save_path, "train_coef.xlsx")
test_coef_file = os.path.join(save_path, "test_coef.xlsx")
merge_data = merge_excel(train_coef_file, test_coef_file)
merge_data.to_excel(os.path.join(save_path, "all_coef.xlsx"), index=False)