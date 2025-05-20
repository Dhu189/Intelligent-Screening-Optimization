import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from itertools import product
from scipy.optimize import minimize
from joblib import Parallel, delayed
import time

start_time = time.time()

data = pd.read_excel("dataset_path.xlsx").fillna(-1)

models = {}
for name in ['DC', 'DL', 'PS', 'TC']:
    model_path = Path("models") / f"{name}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    models[name] = joblib.load(model_path)

TC_vals = np.arange(1.0, 3.0, 0.1)
DC_vals = np.arange(2.5, 4.5, 0.1)
DL_vals = np.arange(0.1, 1.5, 0.1)
PS_vals = np.arange(1.0, 3.0, 0.1)
sampling_combinations = list(product(TC_vals, DC_vals, DL_vals, PS_vals))

def objective_function(features, targets):
    predictions = np.array([
        models[name].predict(features.reshape(1, -1))[0] for name in ['DC', 'DL', 'PS', 'TC']
    ])
    return np.sum((predictions - targets) ** 2)

bounds = [(0, 1)] * 15
def constraint_sum_1(x): return np.sum(x) - 1
def constraint_sum_first3_lower(x): return np.sum(x[:3]) - 0.45
def constraint_sum_first3_upper(x): return 0.7 - np.sum(x[:3])
def make_feature3_constraint(limit): return lambda x: limit - x[2]

def optimize_features(sample, f3_limit):
    targets = np.array([sample[1], sample[2], sample[3], sample[0]])
    initial_guess = np.random.rand(15)
    initial_guess /= np.sum(initial_guess)
    constraints = [
        {'type': 'eq', 'fun': constraint_sum_1},
        {'type': 'ineq', 'fun': constraint_sum_first3_lower},
        {'type': 'ineq', 'fun': constraint_sum_first3_upper},
        {'type': 'ineq', 'fun': make_feature3_constraint(f3_limit)}
    ]
    result = minimize(
        objective_function,
        initial_guess,
        args=(targets,),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'ftol': 1e-9}
    )
    if result.success:
        return list(sample) + list(result.x)
    return None

def apply_model_predictions(df):
    for name in ['DC', 'DL', 'PS', 'TC']:
        df[name] = df[[f'Feature_{i+1}' for i in range(15)]].apply(
            lambda row: models[name].predict(row.values.reshape(1, -1))[0],
            axis=1
        )
    return df

feature_3_limits = [0.20, 0.15, 0.10, 0.05, 0.0]
for f3_limit in feature_3_limits:
    results = Parallel(n_jobs=-1)(
        delayed(optimize_features)(sample, f3_limit) for sample in sampling_combinations
    )
    valid_results = [r for r in results if r is not None]
    columns = ['TC_target', 'DC_target', 'DL_target', 'PS_target'] + [f'Feature_{i+1}' for i in range(15)]
    df = pd.DataFrame(valid_results, columns=columns)
    df = apply_model_predictions(df)
    output_file = f"optimized_results_feature3_max_{f3_limit:.2f}.xlsx"
    df.to_excel(output_file, index=False)

print(f"Completed in {time.time() - start_time:.2f} seconds.")

