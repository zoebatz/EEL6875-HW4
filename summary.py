import os
import pandas as pd
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
  #  default="SAC",
    required=True,
    help="RL model name (SAC, PPO, TD3, DDPG) required."
)
args = parser.parse_args()

model_name = args.model_name

# Path to your hyperparam sweep folder
root_dir = fr"C:\Users\Mitch\Desktop\zoe\EEL6875 HW4\{model_name}_hyperparam_sweep_results" 

# List to store all runs' results
all_runs = []

# Recursively search for all test_metrics CSVs
for csv_file in glob.glob(os.path.join(root_dir, "**", "acc_test_metrics_*.csv"), recursive=True):
    # Example path:
    # ...\chunk100_lr0.0003_batch512_buf500000_steps300000_20251026_210304_chunk100_20251026_210308\test_metrics_chunk100_20251026_210308.csv
    folder = os.path.basename(os.path.dirname(csv_file))

    # ---- SAC hyperparams ----
    if model_name == 'SAC':
        # Parse hyperparameters from folder name
        parts = folder.split("_")
        try:
            chunk = parts[0].replace("chunk", "")
            lr = parts[1].replace("lr", "")
            batch = parts[2].replace("batch", "")
            buf = parts[3].replace("buf", "")   # remove for PPO
            steps = parts[4].replace("steps", "")   # change to parts[3] for PPO
            timestamp = parts[-1]
        except Exception as e:
            print(f"[WARN] Skipping folder (unrecognized format): {folder}")
            continue

        # Load metrics from CSV
        df = pd.read_csv(csv_file)
        df["chunk_size"] = chunk
        df["learning_rate"] = lr
        df["batch_size"] = batch
        df["buffer_size"] = buf
        df["total_timesteps"] = steps
        df["timestamp"] = timestamp
    
    # ---- PPO hyperparams ----
    elif model_name == 'PPO':
        # Parse hyperparameters from folder name
        parts = folder.split("_")
        try:
            chunk = parts[0].replace("chunk", "")
            lr = parts[1].replace("lr", "")
            batch = parts[2].replace("batch", "")
            steps = parts[3].replace("steps", "")   
            gamma = parts[4].replace("gamma", "")
            ent_coeff = parts[5].replace("ent", "")
            timestamp = parts[-1]
        except Exception as e:
            print(f"[WARN] Skipping folder (unrecognized format): {folder}")
            continue

        # Load metrics from CSV
        df = pd.read_csv(csv_file)
        df["chunk_size"] = chunk
        df["learning_rate"] = lr
        df["batch_size"] = batch
        df["total_timesteps"] = steps
        df["gamma"] = gamma
        df["ent_coeff"] = ent_coeff
        df["timestamp"] = timestamp

    # ---- TD3 hyperparams ----
    elif model_name == 'TD3' or 'DDPG':
        # Parse hyperparameters from folder name
        parts = folder.split("_")
        try:
            chunk = parts[0].replace("chunk", "")
            lr = parts[1].replace("lr", "")
            batch = parts[2].replace("batch", "")
            buf = parts[3].replace("buf", "")
            gamma = parts[4].replace("gamma", "")
            tau = parts[5].replace("tau", "")
            steps = parts[6].replace("steps", "")
            timestamp = parts[-1]
        except Exception as e:
            print(f"[WARN] Skipping folder (unrecognized format): {folder}")
            continue

        # Load metrics from CSV
        df = pd.read_csv(csv_file)
        df["chunk_size"] = chunk
        df["learning_rate"] = lr
        df["batch_size"] = batch
        df["gamma"] = gamma
        df["tau"] = tau
        df["total_timesteps"] = steps
        df["timestamp"] = timestamp

    all_runs.append(df)

# Combine all into one DataFrame
summary_df = pd.concat(all_runs, ignore_index=True)

# Sort by RMSE ascending
summary_df = summary_df.sort_values("RMSE_speed")

# Print the best config
print("\n[Best configuration by RMSE]")
print(summary_df.iloc[0])


# Save the combined CSV
output_file = os.path.join(root_dir, f"{model_name}_combined_hyperparam_results.csv")
summary_df.to_csv(output_file, index=False)

print(f"[INFO] Combined {len(all_runs)} runs into {output_file}")
print(summary_df.head())


