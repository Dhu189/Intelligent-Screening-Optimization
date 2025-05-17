import pandas as pd
import numpy as np
import os

file_path = "path_to_your_data/optimized_results.xlsx"
df = pd.read_excel(file_path)
feature_df = df.iloc[:, 4:19].copy()
num_rows = feature_df.shape[0]
to_delete = set()

for i in range(num_rows):
    if i in to_delete:
        continue
    for j in range(i + 1, num_rows):
        if j in to_delete:
            continue
        diff = np.abs(feature_df.iloc[i] - feature_df.iloc[j])
        if np.all(diff < 0.05):
            to_delete.add(j)

df_filtered = df.drop(index=list(to_delete)).reset_index(drop=True)
output_path = os.path.join(os.path.dirname(file_path), "filtered_results.xlsx")
df_filtered.to_excel(output_path, index=False)
print("Deduplication complete. Cleaned data saved to:", output_path)
