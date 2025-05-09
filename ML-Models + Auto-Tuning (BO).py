import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import optuna

# ========== 1. Load Data ==========
file_path = "path_to_your_dataset.xlsx"  # Replace with your actual file path
data = pd.read_excel(file_path).fillna(-1)

# Separate targets (first 4 columns) and features (last 15 columns)
targets = data.iloc[:, :4].values
features = data.iloc[:, 4:].values
target_names = ['DC', 'DL', 'PS', 'TC']


# ========== 2. Hyperparameter Tuning with Optuna ==========
def tune_model(target_idx, n_trials=100):
    target = targets[:, target_idx]

    def objective(trial):
        # Suggest hyperparameters
        hidden_layer_size = trial.suggest_int('hidden_layer_size', 50, 5000)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
        max_iter = trial.suggest_int('max_iter', 50, 5000)
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2)

        # Train model
        model = MLPRegressor(hidden_layer_sizes=(hidden_layer_size,),
                             activation=activation,
                             max_iter=max_iter,
                             alpha=alpha,
                             random_state=42)
        model.fit(features, target)
        preds = model.predict(features)

        # Evaluation metrics
        r2 = r2_score(target, preds)
        mse = mean_squared_error(target, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target, preds)
        medae = median_absolute_error(target, preds)
        mre = np.mean(np.abs((target - preds) / (target + 1e-8)))
        mape = mre * 100

        # Store metrics
        trial.set_user_attr('r2', r2)
        trial.set_user_attr('mse', mse)
        trial.set_user_attr('rmse', rmse)
        trial.set_user_attr('mae', mae)
        trial.set_user_attr('medae', medae)
        trial.set_user_attr('mre', mre)
        trial.set_user_attr('mape', mape)

        return -mse

    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Best result
    best = study.best_trial
    return {
        'params': best.params,
        'r2': best.user_attrs['r2'],
        'mse': best.user_attrs['mse'],
        'rmse': best.user_attrs['rmse'],
        'mae': best.user_attrs['mae'],
        'medae': best.user_attrs['medae'],
        'mre': best.user_attrs['mre'],
        'mape': best.user_attrs['mape']
    }


# ========== 3. Run Optimization ==========
n_trials_list = [100]
results = []

for n_trials in n_trials_list:
    print(f"Running with {n_trials} trials...")
    for idx, name in enumerate(target_names):
        result = tune_model(idx, n_trials)
        results.append({
            'Target': name,
            'Trials': n_trials,
            'Hidden Layer Size': result['params']['hidden_layer_size'],
            'Activation': result['params']['activation'],
            'Max Iter': result['params']['max_iter'],
            'Alpha': result['params']['alpha'],
            'R2': result['r2'],
            'MSE': result['mse'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'MedAE': result['medae'],
            'MRE': result['mre'],
            'MAPE': result['mape']
        })

# ========== 4. Save Results ==========
df_results = pd.DataFrame(results)
df_results.to_excel("model_best_results.xlsx", index=False)
print("Best results saved to 'model_best_results.xlsx'.")
