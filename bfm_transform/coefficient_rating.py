import os
import scipy.io
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import pickle
import cudf

# Set to directory containing optimized_svr_model_*.pkl and scaler_*.pkl
path = "your_location"
try:
    model_name = "optimized_svr_model_5.pkl"
    model_path = os.path.join(path, model_name)
    # Load saved scaler
    scaler_name = "scaler_5.pkl"
    scaler_path = os.path.join(path, scaler_name)
    with open(scaler_path, 'rb') as file:
        loaded_scaler = pickle.load(file)
        
    with open(model_path, 'rb') as file:
        loaded_svr = pickle.load(file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Set to folder containing modify_*_*.txt (e.g. beautify_demo/AF84)
save_path = "your_location"
modify_vector_path = os.path.join(save_path, "modify_AF84_10.txt")
with open(modify_vector_path, "r") as file:
    vector_string = file.read()

# Score the coefficient vector
print("Start rating")
data = eval(vector_string)
data_array = np.array(data).reshape(1, -1)
print(data_array.shape)
# Normalize
data_scaled = loaded_scaler.transform(data_array.astype(float))

# Convert to cuDF DataFrame for prediction
data_cuml = cudf.DataFrame(data_scaled)

# Predict
score_array = loaded_svr.predict(data_cuml).to_numpy()
score = score_array[0]
print(score)