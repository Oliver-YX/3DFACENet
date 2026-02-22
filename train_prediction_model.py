import os
import scipy.io
import optuna
import numpy as np
import random
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
import cudf
from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
from cuml.svm import SVR as cuSVR
from cuml.experimental.linear_model import Lars as cuLars
from cuml.linear_model import ElasticNet as cuElasticNet
from cuml.preprocessing import StandardScaler
from cuml.metrics import mean_squared_error as cu_mean_squared_error
from cuml.metrics import mean_absolute_error as cu_mean_absolute_error
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

storage = "sqlite:///optuna_flame.db"
# optuna-dashboard sqlite:///optuna_flame.db 
save_path = "your_save_path"


def load_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    y = df['score'].to_numpy()
    X = df.drop(columns=['filename', 'score']).to_numpy()
    return X, y

if not os.path.exists(save_path):
    os.makedirs(save_path)

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def harmonic_mean(x, y):
    if x > 0 and y > 0:
        return 2 / (1/x + 1/y)
    else:
        return 0  
    
def add_noise(X, noise_level):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def objective_enet(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    model = cuElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train_cuml, y_train_cuml)
    preds = model.predict(X_test_cuml)
    rmse = cu_mean_squared_error(y_test_cuml, preds, squared=False)
    mae = cu_mean_absolute_error(y_test_cuml, preds)
    
    # record information
    trial.report(mae, step=1) 
    trial.set_user_attr("mae", float(mae))
    trial.set_user_attr("rmse", float(rmse))

    return mae

def objective_rf(trial):
    # rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 200)
    # rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
    # model = cuRandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 300)
    rf_max_depth = trial.suggest_int("rf_max_depth", 3, 30)
    rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 20)
    rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 2, 20)
    
    model = cuRandomForestRegressor(
        n_estimators=rf_n_estimators, 
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf
    )
    model.fit(X_train_cuml, y_train_cuml)
    preds = model.predict(X_test_cuml)
    
    rmse = cu_mean_squared_error(y_test_cuml, preds, squared=False)
    mae = cu_mean_absolute_error(y_test_cuml, preds)        
    
    # record information
    trial.report(mae, step=1) 
    trial.set_user_attr("mae", float(mae))
    trial.set_user_attr("rmse", float(rmse))
    return mae

def objective_lars(trial):
    n_nonzero_coefs = trial.suggest_int('n_nonzero_coefs', 1, X_train_cuml.shape[1])

    model = cuLars(fit_intercept=True, normalize=True, n_nonzero_coefs=n_nonzero_coefs)
    model.fit(X_train_cuml, y_train_cuml)

    preds = model.predict(X_test_cuml)
    rmse = cu_mean_squared_error(y_test_cuml, preds, squared=False)
    mae = cu_mean_absolute_error(y_test_cuml, preds)
    
    # record information
    trial.report(mae, step=1) 
    trial.set_user_attr("mae", float(mae))
    trial.set_user_attr("rmse", float(rmse))

    return rmse

def objective_svr(trial):
    svr_c = trial.suggest_float("svr_c", 1e-5, 1e2, log=True)
    svr_epsilon = trial.suggest_float("svr_epsilon", 1e-3, 1e0)
    # svr_kernel = trial.suggest_categorical("svr_kernel", ["poly", "rbf", "sigmoid"])

    # # build model
    # svr_degree = trial.suggest_int("svr_degree", 1, 4)
    # svr_coef0 = trial.suggest_float("svr_coef0", 0, 5)
    svr_gamma = trial.suggest_float("svr_gamma", 1e-6, 1e-1, log=True)

    model = cuSVR(C=svr_c, epsilon=svr_epsilon, kernel='rbf', gamma=svr_gamma)

    model.fit(X_train_cuml, y_train_cuml)
    preds = model.predict(X_test_cuml)
    rmse = cu_mean_squared_error(y_test_cuml, preds, squared=False)
    mae = cu_mean_absolute_error(y_test_cuml, preds)
    har = harmonic_mean(mae, rmse)

    # record information
    trial.report(mae, step=1) 
    trial.set_user_attr("mae", float(mae))
    trial.set_user_attr("rmse", float(rmse))
    trial.set_user_attr("har", float(har))
    return mae 
    
    
def objective_xgb(trial):
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
        "max_depth": trial.suggest_int("max_depth", 1, 30),
        "learning_rate": trial.suggest_float("learning_rate", 1e-10, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "lambda": trial.suggest_float("lambda", 1e-5, 15, log=True),
        "alpha": trial.suggest_float("alpha", 1e-5, 10, log=True),
        "tree_method": "hist",
        "device": "cuda"
    }
    model = xgb.XGBRegressor(**param)
    model.fit(X_train_cuml, y_train_cuml)
    preds = model.predict(X_test_cuml)    
    rmse = cu_mean_squared_error(y_test_cuml, preds, squared=False)
    mae = cu_mean_absolute_error(y_test_cuml, preds)        

    # record information
    trial.report(mae, step=1) 
    trial.set_user_attr("mae", float(mae))
    trial.set_user_attr("rmse", float(rmse))
    return mae

def load_fold_data(fold_number, base_directory):
    train_file = f'{base_directory}/cross_validation_{fold_number}/train_{fold_number}.txt'
    test_file = f'{base_directory}/cross_validation_{fold_number}/test_{fold_number}.txt'
    X_train, y_train = load_data(train_file, directory)
    X_test, y_test = load_data(test_file, directory)
    return X_train, y_train, X_test, y_test


def load_data(file_path, directory):
    X, y = [], []
    with open(file_path, 'r') as file:
        for line in file:
            file_name, score = line.strip().split()
            mat_file = os.path.join(directory, file_name.replace('.jpg', '.mat'))
            if os.path.exists(mat_file):
                mat_data = scipy.io.loadmat(mat_file)
                concatenated_vector = np.concatenate([
                    mat_data["mica_shape"].flatten()
                ])
                X.append(concatenated_vector)
                y.append(float(score))
    return np.array(X), np.array(y)

loss_path = f"{save_path}/flame_trainloss.txt"

# set seed
seed = 42
seed_everything(seed)

# path setting
directory = "your_reconstruction_mat_filepath/"
train_file = 'SCUT-FBP5500-Database-Release/train_test_files/split_of_60%training and 40%testing/train.txt'
test_file = 'SCUT-FBP5500-Database-Release/train_test_files/split_of_60%training and 40%testing/test.txt'

# load data
YOUR_COEF_ROOT = "your_coefficient_excel_directory"
all_coef_path = os.path.join(YOUR_COEF_ROOT, "all_coef.xlsx")
train_coef_path = os.path.join(YOUR_COEF_ROOT, "train_coef.xlsx")
test_coef_path = os.path.join(YOUR_COEF_ROOT, "test_coef.xlsx")

if os.path.exists(train_coef_path) and os.path.exists(test_coef_path):
    print("Found train_coef.xlsx, loading data from Excel...")
    X_train, y_train = load_from_excel(train_coef_path)

    print("Found test_coef.xlsx, loading data from Excel...")
    X_test, y_test = load_from_excel(test_coef_path)
else:
    print("all_coef.xlsx not found, loading data from .mat files...")
    X_train, y_train = load_data(train_file, directory)
    X_test, y_test = load_data(test_file, directory)


n_components = 80  # keep the first 80 components
noise_level = 0    # 1e-5   
X_train_noisy = add_noise(X_train, noise_level)

# normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_noisy)
# save scaler
with open(os.path.join(save_path,'flame_scaler.pkl'), 'wb') as file:
    pickle.dump(scaler, file)
X_test_scaled = scaler.transform(X_test)

# apply PCA
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# convert to cuDF DataFrame
X_train_cuml = cudf.DataFrame(X_train_pca)
y_train_cuml = cudf.Series(y_train)

X_test_cuml = cudf.DataFrame(X_test_pca)
y_test_cuml = cudf.Series(y_test)

####################################################################################################

# # optimize SVR
study_svr = optuna.create_study(study_name = "ijcai_svr2", storage=storage, direction="minimize", load_if_exists=True)
study_svr.optimize(objective_svr, n_trials=10)
best_params_svr = study_svr.best_params
best_result_svr = study_svr.best_value

print(best_params_svr)

optimized_svr = cuSVR(C=best_params_svr["svr_c"], 
                          epsilon=best_params_svr["svr_epsilon"], 
                          kernel="rbf")

# train model
optimized_svr.fit(X_train_cuml, y_train_cuml)

# save model
save_model(optimized_svr, os.path.join(save_path, 'flame_optimized_svr_model.pkl'))

# predict
svr_predictions = optimized_svr.predict(X_test_cuml)

svr_rmse = cu_mean_squared_error(y_test_cuml, svr_predictions, squared=False)
svr_mae = cu_mean_absolute_error(y_test_cuml, svr_predictions)
svr_pc, _ = pearsonr(y_test_cuml.to_numpy(), svr_predictions.to_numpy())

# save results
with open(loss_path, 'a') as file:
    file.write("SVR Performance with Optimized Parameters\n")
    file.write(f"Best Parameters: {best_params_svr}\n")
    file.write(f"Best Result: {best_result_svr}\n")
    file.write("RMSE \t MAE \t PC\n")
    file.write(f"{svr_rmse:.4f} \t {svr_mae:.4f} \t {svr_pc:.4f}\n\n")
print(f"test RMSE:{svr_rmse:.4f} \t MAE:{svr_mae:.4f}\t PC:{svr_pc:.4f}\n")  
