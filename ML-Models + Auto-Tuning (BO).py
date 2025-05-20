import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
)
import optuna
import os

data = pd.read_excel("your_dataset_path.xlsx").fillna(-1)
target = data.iloc[:, 1].values
features = data.iloc[:, 4:].values

output_dir = "your_output_directory"
os.makedirs(output_dir, exist_ok=True)

def tune_model_DL(n_trials):
    def objective(trial):
        hidden_layer_size = trial.suggest_int('hidden_layer_size', 50, 5000)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
        max_iter = trial.suggest_int('max_iter', 50, 5000)
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2)

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        model = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_size,),
            activation=activation,
            max_iter=max_iter,
            alpha=alpha,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        medae = median_absolute_error(y_test, preds)
        mre = np.mean(np.abs((y_test - preds) / (y_test + 1e-8)))
        mape = mre * 100

        trial.set_user_attr('r2', r2)
        trial.set_user_attr('mse', mse)
        trial.set_user_attr('rmse', rmse)
        trial.set_user_attr('mae', mae)
        trial.set_user_attr('medae', medae)
        trial.set_user_attr('mre', mre)
        trial.set_user_attr('mape', mape)

        return r2

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_trial = study.best_trial
    best_params = best_trial.params

    return {
        'Target': 'DL',
        'Trials': n_trials,
        'Best Hidden Layer Size': best_params['hidden_layer_size'],
        'Best Activation': best_params['activation'],
        'Best Max Iter': best_params['max_iter'],
        'Best Alpha': best_params['alpha'],
        'R2': best_trial.user_attrs['r2'],
        'MSE': best_trial.user_attrs['mse'],
        'RMSE': best_trial.user_attrs['rmse'],
        'MAE': best_trial.user_attrs['mae'],
        'MedAE': best_trial.user_attrs['medae'],
        'MRE': best_trial.user_attrs['mre'],
        'MAPE': best_trial.user_attrs['mape']
    }

search_trials_list = [100, 300, 500, 700, 900, 1200, 1500, 1700, 2000]

for n_trials in search_trials_list:
    result = tune_model_DL(n_trials)
    df_result = pd.DataFrame([result])
    output_file = os.path.join(output_dir, f"DL_model_results_trials_{n_trials}.xlsx")
    df_result.to_excel(output_file, index=False)
