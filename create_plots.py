import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, TD3, DDPG
from RL_assignment import TestEnv, DATA_LEN, MODEL_DICT
import os
import pandas as pd



# ---- modify for model and path ----
model_name = 'SAC'

#timestamp = '20251029_235936'
log_dir = 'sac_reward_modes\logs_chunk_training_SAC_huber_20251031_203415'
split = log_dir.split("_")
timestamp = f'{split[-2]}_{split[-1]}'
#'.\hyperparam_sweep_results/chunk200_lr0.0003_batch256_buf500000_steps100000_20251027_211100_chunk200_20251027_211105'

# load trained model
model_path = os.path.join(log_dir, f'{model_name}_speed_follow.zip')
model = MODEL_DICT[model_name].load(model_path)

log_path = os.path.join(log_dir, f"training_log_{timestamp}.csv")
df_log = pd.read_csv(log_path)

# load data
CSV_FILE = "speed_profile.csv"
df = pd.read_csv(CSV_FILE)
full_speed_data = df["speed"].values

# receate test env
test_env = TestEnv(full_speed_data, delta_t=1.0)

obs, _ = test_env.reset()
predicted_speeds = []
reference_speeds = []
rewards = []
errors = []

for _ in range(DATA_LEN):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    predicted_speeds.append(obs[0])  # current_speed
    reference_speeds.append(obs[1])  # reference_speed
    rewards.append(reward)
    errors.append(abs(obs[0] - obs[1]))
    if terminated or truncated:
        break

# compute metrics
mae = np.mean(errors)
rmse = np.sqrt(np.mean(np.square(errors)))
print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")

corr = np.corrcoef(predicted_speeds, reference_speeds)[0, 1]

# plot
plt.figure(figsize=(10, 5))
plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
plt.plot(predicted_speeds, label="Predicted Speed")
plt.title("Speed Following Test (Reloaded Model)")
plt.xlabel("Timestep")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(errors, color="red", alpha=0.7)
plt.title("Speed Error over Time")
plt.xlabel("Timestep")
plt.ylabel("Absolute Error (m/s)")
plt.show()



# predicted vs reference scatter plot
plt.figure(figsize=(5, 5))
plt.scatter(reference_speeds, predicted_speeds, alpha=0.4, color='orange', edgecolor='k')
plt.plot([0, 25], [0, 25], 'k--', label='Ideal Tracking')
plt.xlabel('Reference Speed (m/s)')
plt.ylabel('Predicted Speed (m/s)')
plt.title(f'Predicted vs Reference Speeds (corr={corr:.2f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, f"pred_vs_ref_scatter.png"))
plt.show()

# error vs reference speed
plt.figure(figsize=(6, 4))
plt.scatter(reference_speeds, errors, alpha=0.4, color='red')
plt.title('Absolute Error vs. Reference Speed')
plt.xlabel('Reference Speed (m/s)')
plt.ylabel('Absolute Error (m/s)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, f"error_vs_ref_scatter.png"))
plt.show()

# plot convergence
# Smooth the reward curve (optional)
df_log['smoothed_reward'] = df_log['average_reward'].rolling(window=10, min_periods=1).mean()

# Plot convergence
plt.figure(figsize=(8,4))
plt.plot(df_log['timestep'], df_log['average_reward'], color='lightgray', label='Raw')
plt.plot(df_log['timestep'], df_log['smoothed_reward'], color='blue', linewidth=2, label='Smoothed (window=10)')
plt.xlabel("Training Timestep")
plt.ylabel("Average Episode Reward")
plt.title("Training Convergence (Average Reward vs Timestep)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "training_convergence.png"))
plt.show()