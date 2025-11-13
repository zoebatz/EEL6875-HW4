import os
import pandas as pd
import glob

# Path to your hyperparam sweep folder
root_dir = r"C:\Users\Mitch\Desktop\zoe\EEL6875 HW4\SAC_hyperparam_sweep_results"

model_name = 'SAC'
# List to store all runs' results
all_runs = []

# Recursively search for all test_metrics CSVs
for csv_file in glob.glob(os.path.join(root_dir, "**", "acc_test_metrics_*.csv"), recursive=True):
    # Example path:
    # ...\chunk100_lr0.0003_batch512_buf500000_steps300000_20251026_210304_chunk100_20251026_210308\test_metrics_chunk100_20251026_210308.csv
    folder = os.path.basename(os.path.dirname(csv_file))

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


