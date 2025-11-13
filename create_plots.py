import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, TD3, DDPG
from RL_assignment import TestEnv, DATA_LEN, MODEL_DICT
import os
import pandas as pd
import csv


# ---- modify for model and path ----
model_name = 'SAC'

#timestamp = '20251029_235936'
log_dir = 'logs_chunk_training_SAC_20251112_175719'
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


test_env = TestEnv(
        full_data=full_speed_data,
        delta_t=1.0,   # match training dt
        d_init=20.0,   # start mid-band
        d_min=5.0, d_max=30.0
    )

# reset environment for full test
obs, _ = test_env.reset()
v_ego_list, v_lead_list, d_list, j_list = [], [], [], []
rewards, actions = [], []

for _ in range(DATA_LEN):
    action, _ = model.predict(obs, deterministic=True)  # no exploration in testing
    obs, reward, terminated, truncated, info = test_env.step(action)

    # log everything for eval
    # obs = [v_ego, v_lead, d, v_rel, a_prev]
    v_ego, v_lead, d, _, _ = obs
    v_ego_list.append(float(v_ego))
    v_lead_list.append(float(v_lead))
    d_list.append(float(d))
    j_list.append(float(info["jerk"]))
    rewards.append(float(reward))
    actions.append(float(np.clip(action[0], -2.0, 2.0)))

    if terminated or truncated:
        break

# convert to arrays for easier computing 
v_ego_arr = np.array(v_ego_list)
v_lead_arr = np.array(v_lead_list)
d_arr = np.array(d_list)
j_arr = np.array(j_list)

# metrics for report
speed_diff = np.abs(v_ego_arr - v_lead_arr)
mae_speed = float(np.mean(speed_diff))
rmse_speed = float(np.sqrt(np.mean((v_ego_arr - v_lead_arr) ** 2)))
jerk_mean = float(np.mean(j_arr))
jerk_var = float(np.var(j_arr))
in_range_pct = float(np.mean((d_arr >= 5.0) & (d_arr <= 30.0)) * 100.0)
avg_test_reward = float(np.mean(rewards))
corr = float(np.corrcoef(v_ego_arr, v_lead_arr)[0, 1])

print(f"[TEST] MAE={mae_speed:.3f}, RMSE={rmse_speed:.3f}, "
    f"Jerk mean={jerk_mean:.3f}, Jerk var={jerk_var:.3f}, "
    f"InRange%={in_range_pct:.1f}, Corr={corr:.3f}")

# save metrics
metrics_path = os.path.join(log_dir, f"acc_test_metrics_{timestamp}.csv")
with open(metrics_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["avg_test_reward", "MAE_speed", "RMSE_speed",
                    "jerk_mean", "jerk_var", "in_range_pct", "Corr Coeff"])
    writer.writerow([avg_test_reward, mae_speed, rmse_speed,
                    jerk_mean, jerk_var, in_range_pct, corr])

# plots
# 1) Ego vs Lead speed
plt.figure(figsize=(10, 5))
plt.plot(v_lead_arr, label="Lead Speed", linestyle="--")
plt.plot(v_ego_arr, label="Ego Speed", linestyle="-")
plt.xlabel("Timestep"); plt.ylabel("Speed (m/s)")
plt.title(f"ACC: Ego vs Lead Speeds)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(log_dir, f"acc_speed_tracking_{timestamp}.png"))

# 2) Distance over time with safe band [5,30] m
plt.figure(figsize=(10, 4))
plt.plot(d_arr, label="Following Distance")
plt.axhspan(5.0, 30.0, alpha=0.15, label="Safe Range [5,30] m")
plt.xlabel("Timestep"); plt.ylabel("Distance (m)")
plt.title(f"Following Distance Over Time (In Range Percentage={in_range_pct:.1f})")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(log_dir, f"acc_distance_over_time_{timestamp}.png"))

# 3) Jerk over time
plt.figure(figsize=(10, 4))
plt.plot(j_arr)
plt.xlabel("Timestep"); plt.ylabel("Jerk (m/s³)")
plt.title(f"Jerk Over Time (mean={jerk_mean:.3f}, var={jerk_var:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, f"acc_jerk_over_time_{timestamp}.png"))

# 4) Speed difference |v_ego - v_lead|
plt.figure(figsize=(10, 4))
plt.plot(speed_diff)
plt.xlabel("Timestep"); plt.ylabel("|Δv| (m/s)")
plt.title(f"Speed Difference (MAE={mae_speed:.3f}, RMSE={rmse_speed:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, f"acc_speed_diff_{timestamp}.png"))



'''
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
'''