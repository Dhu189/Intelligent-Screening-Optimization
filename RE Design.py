import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from itertools import product
from scipy.optimize import minimize
from joblib import Parallel, delayed
import time

start_time = time.time()

file_path = Path("path_to_dataset.xlsx")
data = pd.read_excel(file_path).fillna(-1)

TC_vals = np.arange(1.0, 3.0, 0.1)
DC_vals = np.arange(2.5, 4.5, 0.1)
DL_vals = np.arange(0.1, 1.5, 0.1)
PS_vals = np.arange(1.0, 3.0, 0.1)

sampling_combinations = list(product(TC_vals, DC_vals, DL_vals, PS_vals))
print(f"Total generated samples: {len(sampling_combinations)}")

model_dir = Path("models")
models = {}
for name in ['DC', 'DL', 'PS', 'TC']:
    model_path = model_dir / f"{name}_model.pkl"
    if model_path.exists():
        models[name] = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model for {name} not found: {model_path}")

def objective_function(features, target_values):
    predictions = np.array([models[name].predict(features.reshape(1, -1))[0] for name in ['DC', 'DL', 'PS', 'TC']])
    return np.sum((predictions - target_values) ** 2)

bounds = [(0, 1)] * 15

def constraint_sum_1(x): return np.sum(x) - 1
def constraint_sum_1_2_3(x): return np.sum(x[:3]) - 0.45
def constraint_sum_1_2_3_upper(x): return 0.7 - np.sum(x[:3])

def make_constraint_feature_3_max(limit):
    def constraint(x): return limit - x[2]
    return constraint

def optimize_features(sample, f3_limit):
    tc, dc, dl, ps = sample
    targets = np.array([dc, dl, ps, tc])
    initial_guess = np.random.rand(15)
    initial_guess /= np.sum(initial_guess)

    constraints = [
        {'type': 'eq', 'fun': constraint_sum_1},
        {'type': 'ineq', 'fun': constraint_sum_1_2_3},
        {'type': 'ineq', 'fun': constraint_sum_1_2_3_upper},
        {'type': 'ineq', 'fun': make_constraint_feature_3_max(f3_limit)}
    ]

    result = minimize(objective_function, initial_guess, args=(targets,),
                      bounds=bounds, constraints=constraints, method='SLSQP',
                      options={'ftol': 1e-9})

    if result.success:
        return [tc, dc, dl, ps] + list(result.x)
    return None

feature_3_limits = [0.20, 0.15, 0.10, 0.05, 0.0]

for f3_limit in feature_3_limits:
    print(f"Running optimization with Feature_3 ≤ {f3_limit}...")

    results = Parallel(n_jobs=18)(
        delayed(optimize_features)(sample, f3_limit) for sample in sampling_combinations
    )
    valid_results = [r for r in results if r is not None]

    columns = ['TC', 'DC', 'DL', 'PS'] + [f'Feature_{i+1}' for i in range(15)]
    df_results = pd.DataFrame(valid_results, columns=columns)
    output_file = f"optimized_results_feature3_max_{f3_limit:.2f}.xlsx"
    df_results.to_excel(output_file, index=False)

    print(f"Saved {len(valid_results)} results to {output_file}")

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")
