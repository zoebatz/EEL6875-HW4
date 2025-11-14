'''import pandas as pd

input_csv = '.\hyperparam_sweep_results\combined_hyperparam_results.csv'
output_csv = '.\hyperparam_sweep_results\cleaned_hyperparam_results.csv'

df = pd.read_csv(input_csv)


# determine unique runs
cols = ['chunk_size', 'learning_rate', 'batch_size', 'buffer_size', 'total_timesteps']

df_unique = df.drop_duplicates(subset=cols, keep='first')
df_unique.to_csv(output_csv, index=False)

'''

import os

root_dir = r"C:/Users/Mitch/Desktop/zoe/EEL6875 HW4/PPO_hyperparam_sweep_results"

for folder in os.listdir(root_dir):
    old_path = os.path.join(root_dir, folder)

    # Only rename folders
    if not os.path.isdir(old_path):
        continue

    # Only rename if they contain the old token
    if "ent_coeff" in folder:
        new_folder = folder.replace("ent_coeff", "ent")
        new_path = os.path.join(root_dir, new_folder)

        # Perform rename
        os.rename(old_path, new_path)

        print(f"Renamed:\n  {folder}\n→ {new_folder}")

