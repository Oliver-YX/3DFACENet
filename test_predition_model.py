import os
import scipy.io
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from coefficient_score_network import ModifiedResNet1DModel,Transformer
from math import sqrt
import pickle
import cudf

path = "your_model_and_data_path"
save_path = "your_save_path"
excel_path = f"{save_path}/beaut_train.xlsx"
loss_file_path = f"{save_path}/train_result.txt"

if not os.path.exists(save_path):
    os.makedirs(save_path)

try:
    model_name = "optimized_svr_model_5.pkl"
    model_path = f"{path}/{model_name}"
    # load scaler   
    scaler_name = "scaler_5.pkl"
    scaler_path = f"{path}/{scaler_name}"
    with open(scaler_path, 'rb') as file:
        loaded_scaler = pickle.load(file)
        
    with open(model_path, 'rb') as file:
        loaded_svr = pickle.load(file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    
    
    
directory = 'your_testdata_path'

df = pd.read_excel(directory)
results = []
print("Start rating")
for index, row in df.iterrows():
    image_name = row[0]
    gt_score = row[1]
    
    # load 3D face coefficient
    data = row[2:162].values.reshape(1, -1)

    # normalize
    data_scaled = loaded_scaler.transform(data.astype(float))

    # convert to cuDF DataFrame
    data_cuml = cudf.DataFrame(data_scaled)

    # predict
    score_array = loaded_svr.predict(data_cuml).to_numpy()
    score = score_array[0]
    
    # save file name, ground truth score and predicted score
    results.append((image_name, gt_score, score))


# save to Excel
df = pd.DataFrame(results, columns=['Name', 'GT', 'Prediction'])
df.to_excel(excel_path, index=False)


mae = mean_absolute_error(df['GT'], df['Prediction'])
rmse =np.sqrt(mean_squared_error(df['GT'], df['Prediction']))
pearson_corr, _ = pearsonr(df['GT'], df['Prediction'])
with open(loss_file_path, 'w') as file:
    file.write(f"model name:{model_name}\n")
    file.write(f"PC:{pearson_corr}\tMAE:{mae}\tRMSE:{rmse}\n")


print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"Pearson Correlation Coefficient: {pearson_corr}")

print(f"Scores saved to {excel_path}")
