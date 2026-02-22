import os
import scipy.io
import optuna
import numpy as np
import random
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
# from cuml.model_selection import train_test_split,KFold
from cuml.preprocessing import StandardScaler
from cuml.metrics import mean_squared_error as cu_mean_squared_error
from cuml.metrics import mean_absolute_error as cu_mean_absolute_error
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

storage = "sqlite:///optuna_5folds(flame).db"
# optuna-dashboard sqlite:///optuna_5folds.db 
save_path = "your_save_path"

seed = 43
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
    seed_everything(seed)
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def objective_svr(trial):
    svr_c = trial.suggest_float("svr_c", 5e-5, 5e0, log=True)
    svr_epsilon = trial.suggest_float("svr_epsilon", 1e-5, 5e-1)
    # svr_kernel = trial.suggest_categorical("svr_kernel", ["poly", "rbf", "sigmoid"])
    # svr_kernel = trial.suggest_categorical("svr_kernel", ["poly", "rbf"])
    svr_kernel = trial.suggest_categorical("svr_kernel", ["rbf"])
    # svr_degree = trial.suggest_int("svr_degree", 1, 3)
    # svr_coef0 = trial.suggest_float("svr_coef0", 0, 3)
    svr_gamma = trial.suggest_float("svr_gamma", 5e-5, 1e0, log=True)

    results = []
    max_mae = 0
    mean_mae = 0
    min_mae = 1
    max_rmse = 0
    min_pc = 1
    
    # build model
    model = cuSVR(C=svr_c, epsilon=svr_epsilon, kernel=svr_kernel, gamma=svr_gamma)
    
    for fold in range(1, 6):
        seed_everything(seed)
        # load data
        X_train, y_train, X_test, y_test = load_fold_data(fold, cross_val_base_dir)
        
        # add noise and normalize
        noise_level = 1e-5
        X_train_noisy = add_noise(X_train, noise_level)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_noisy)
        X_test_scaled = scaler.transform(X_test)

        # convert to cuDF format
        X_train_cuml = cudf.DataFrame(X_train_scaled)
        y_train_cuml = cudf.Series(y_train)
        X_test_cuml = cudf.DataFrame(X_test_scaled)
        y_test_cuml = cudf.Series(y_test)

        
        model.fit(X_train_cuml, y_train_cuml)

        # evaluation
        preds = model.predict(X_test_cuml)
        mae = cu_mean_absolute_error(y_test_cuml, preds)
        rmse = cu_mean_squared_error(y_test_cuml, preds, squared=False)
        pc, _ = pearsonr(y_test_cuml.to_numpy(), preds.to_numpy())
        max_mae = max(max_mae, mae)
        min_mae = min(min_mae, mae)
        max_rmse = max(max_rmse, rmse)
        min_pc = min(min_pc, pc)
        results.append((fold, rmse, mae, pc))
    
    har = harmonic_mean(max_mae,max_rmse)
    total_rmse, total_mae, total_pc = 0, 0, 0
    num_folds = len(results)
    for _, rmse, mae, pc in results:
        total_rmse += rmse
        total_mae += mae
        total_pc += pc

    mean_rmse = total_rmse / num_folds
    mean_mae = total_mae / num_folds
    mean_pc = total_pc / num_folds
    
    # record information
    trial.report(mean_mae, step=1) 
    trial.set_user_attr("min mae", float(min_mae))
    trial.set_user_attr("max mae", float(max_mae))
    trial.set_user_attr("mean mae", float(mean_mae))
    trial.set_user_attr("rmse", float(max_rmse))
    trial.set_user_attr("har", float(har))
    trial.set_user_attr("pc", float(min_pc))

    return mean_mae

    
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

loss_path = f"{save_path}/trainloss.txt"

####################################################################################################
### 5 folds training process
results = []
seed_everything(seed)
# directory = "your_location/5500_AlbedoGAN/croase/mat"
directory = "your_location_MICA_result_mat"
cross_val_base_dir = "your_location/SCUT-FBP5500-Database-Release/train_test_files/5_folders_cross_validations_files"

study_svr = optuna.create_study(study_name = f"rating_flame", storage=storage, direction="minimize", load_if_exists=True)
study_svr.optimize(objective_svr, n_trials=100)
best_params_svr = study_svr.best_params
best_result_svr = study_svr.best_value

with open(loss_path, 'w') as file:
    file.write("SVR Performance with Optimized Parameters\n")
    file.write(f"Best Parameters: {best_params_svr}\n")
    file.write(f"Best Result: {best_result_svr}\n")
    
for fold in range(1, 6):
    seed_everything(seed)
    print(f"Processing fold {fold}...")
    # load data
    X_train, y_train, X_test, y_test = load_fold_data(fold, cross_val_base_dir)
    noise_level = 1e-5 # 0.00005
    X_train_noisy = add_noise(X_train, noise_level)
    
    # normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_noisy)
    # save scaler
    with open(os.path.join(save_path,f'scaler_{fold}.pkl'), 'wb') as file:
        pickle.dump(scaler, file)
    X_test_scaled = scaler.transform(X_test)

    # convert to cuDF format
    X_train_cuml = cudf.DataFrame(X_train_scaled)
    y_train_cuml = cudf.Series(y_train)
    X_test_cuml = cudf.DataFrame(X_test_scaled)
    y_test_cuml = cudf.Series(y_test)

    # train model
    optimal_kernel = best_params_svr["svr_kernel"]

    if optimal_kernel == "poly":
        optimized_svr = cuSVR(C=best_params_svr["svr_c"], 
                            epsilon=best_params_svr["svr_epsilon"], 
                            kernel=optimal_kernel,
                            degree=best_params_svr["svr_degree"],  
                            coef0=best_params_svr["svr_coef0"])
    elif optimal_kernel == "sigmoid":
        optimized_svr = cuSVR(C=best_params_svr["svr_c"], 
                            epsilon=best_params_svr["svr_epsilon"], 
                            kernel=optimal_kernel,
                            coef0=best_params_svr["svr_coef0"])
    elif optimal_kernel == "rbf":
        optimized_svr = cuSVR(C=best_params_svr["svr_c"], 
                            epsilon=best_params_svr["svr_epsilon"], 
                            kernel=optimal_kernel,
                            gamma=best_params_svr["svr_gamma"])
    else:
        optimized_svr = cuSVR(C=best_params_svr["svr_c"], 
                            epsilon=best_params_svr["svr_epsilon"], 
                            kernel=optimal_kernel)
        
    optimized_svr.fit(X_train_cuml, y_train_cuml)

    # save model
    save_model(optimized_svr, os.path.join(save_path, f'optimized_svr_model_{fold}.pkl'))
    # predict and evaluate
    svr_predictions = optimized_svr.predict(X_test_cuml)
    svr_rmse = cu_mean_squared_error(y_test_cuml, svr_predictions, squared=False)
    svr_mae = cu_mean_absolute_error(y_test_cuml, svr_predictions)
    svr_pc, _ = pearsonr(y_test_cuml.to_numpy(), svr_predictions.to_numpy())
    print(svr_rmse)
    results.append((fold, svr_rmse, svr_mae, svr_pc))

# save results
total_rmse, total_mae, total_pc = 0, 0, 0
num_folds = len(results)
for _, rmse, mae, pc in results:
    total_rmse += rmse
    total_mae += mae
    total_pc += pc

avg_rmse = total_rmse / num_folds
avg_mae = total_mae / num_folds
avg_pc = total_pc / num_folds

with open(loss_path, 'a') as file:
    file.write("Fold\tRMSE\t\tMAE\t\tPC\n")
    for fold, rmse, mae, pc in results:
        file.write(f"{fold}\t {rmse:.4f}\t {mae:.4f}\t {pc:.4f}\n")
    file.write("Average RMSE\tMAE\tPC\n")
    file.write(f"{avg_rmse:.4f}\t {avg_mae:.4f}\t {avg_pc:.4f}\n")
