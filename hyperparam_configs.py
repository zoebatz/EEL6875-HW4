import os
import subprocess
import itertools
import time


# ---- choose model name ----
model_name = 'SAC'

# hyperparameters to test
'''chunk_sizes = [10, 50, 100, 200, 300, 400]
learning_rates = [3e-4, 1e-4]
batch_sizes = [256, 512]
buffer_sizes = [500_000, 1_000_000, 2_000_000]
total_timesteps = [100_000, 300_000, 500_000, 1_000_000]
'''
chunk_sizes = [50, 100, 400]
learning_rates = [3e-4]
batch_sizes = [256, 512]
buffer_sizes = [500_000, 1_000_000, 2_000_000] # remove for PPO
total_timesteps = [100_000, 500_000]

# create all combos
param_grid = list(itertools.product(chunk_sizes, learning_rates, batch_sizes, buffer_sizes, total_timesteps))
print(f"[INFO] Total combinations to run: {len(param_grid)}")

# create folder for all sweep results
os.makedirs(f"{model_name}_hyperparam_sweep_results", exist_ok=True)

# loop through all combos
for i, (chunk, lr, batch, buf, steps) in enumerate(param_grid):
    print(f"\n=== Run {i+1}/{len(param_grid)} ===")
    print(f"chunk={chunk}, lr={lr}, batch={batch}, buffer={buf}, steps={steps}")

    # Unique folder name for this configuration
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"chunk{chunk}_lr{lr}_batch{batch}_buf{buf}_steps{steps}"
    out_dir = os.path.join(f"{model_name}_hyperparam_sweep_results", run_name)

    # Environment variables for RL_assignment.py
    env = os.environ.copy()
    env["LR"] = str(lr)
    env["BATCH_SIZE"] = str(batch)
    env["BUFFER_SIZE"] = str(buf)
    env["TOTAL_STEPS"] = str(steps)



    # Run RL_assignment.py as a subprocess
    cmd = [
        "python", "RL_assignment.py",
        "--chunk_size", str(chunk),
        "--output_dir", out_dir
    ]
    subprocess.run(cmd, env=env)