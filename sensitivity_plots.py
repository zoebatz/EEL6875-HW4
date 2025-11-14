import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import csv

# SAC sensitivity plots
logdir_sac = "SAC_hyperparam_sweep_results/SAC_combined_hyperparam_results.csv"
df_sac = pd.read_csv(logdir_sac)

plt.style.use("ggplot")

# Aggregate RMSE by each hyperparameter
df_chunk = df_sac.groupby("chunk_size")["RMSE_speed"].mean().reset_index()
df_batch = df_sac.groupby("batch_size")["RMSE_speed"].mean().reset_index()
df_gamma = df_sac.groupby("gamma")["RMSE_speed"].mean().reset_index()
df_tau   = df_sac.groupby("tau")["RMSE_speed"].mean().reset_index()

# Create combined figure (wide rectangle)
fig, axes = plt.subplots(2, 2, figsize=(14, 6))

def plot_sensitivity(ax, df, x, title):
    sns.lineplot(data=df, x=x, y="RMSE_speed", marker="o", ci=None, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x)
    ax.set_ylabel("RMSE_speed")
    ax.grid(True)

# Row 1
plot_sensitivity(axes[0, 0], df_chunk, "chunk_size", "SAC: RMSE vs chunk_size")
plot_sensitivity(axes[0, 1], df_batch, "batch_size", "SAC: RMSE vs batch_size")

# Row 2
plot_sensitivity(axes[1, 0], df_gamma, "gamma", "SAC: RMSE vs gamma")
plot_sensitivity(axes[1, 1], df_tau,   "tau",   "SAC: RMSE vs tau")

plt.tight_layout()

# Save in the same folder as the CSV
save_dir = os.path.dirname(logdir_sac)
plt.savefig(os.path.join(save_dir, "SAC_sensitivity_plots.png"))

plt.show()


'''# SAC sensitivity plots
logdir_sac = "SAC_hyperparam_sweep_results/SAC_combined_hyperparam_results.csv"
df_sac = pd.read_csv(logdir_sac)

plt.style.use("ggplot")

# Aggregate RMSE by each hyperparameter
df_chunk   = df_sac.groupby("chunk_size")["RMSE_speed"].mean().reset_index()
df_batch   = df_sac.groupby("batch_size")["RMSE_speed"].mean().reset_index()
df_buffer  = df_sac.groupby("buffer_size")["RMSE_speed"].mean().reset_index()      # change to 'buffer' if needed
df_steps   = df_sac.groupby("total_timesteps")["RMSE_speed"].mean().reset_index()

# Create combined figure (wide rectangle: 2 rows x 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(14, 6))

def plot_sensitivity(ax, df, x, title):
    sns.lineplot(data=df, x=x, y="RMSE_speed", marker="o", ci=None, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x)
    ax.set_ylabel("RMSE_speed")
    ax.grid(True)

# Row 1
plot_sensitivity(axes[0, 0], df_chunk,  "chunk_size",      "SAC: RMSE vs chunk_size")
plot_sensitivity(axes[0, 1], df_batch,  "batch_size",      "SAC: RMSE vs batch_size")

# Row 2
plot_sensitivity(axes[1, 0], df_buffer, "buffer_size",     "SAC: RMSE vs buffer_size")
plot_sensitivity(axes[1, 1], df_steps,  "total_timesteps", "SAC: RMSE vs total_timesteps")

plt.tight_layout()

# Save in the same folder as the CSV
save_dir = os.path.dirname(logdir_sac)
plt.savefig(os.path.join(save_dir, "SAC_sensitivity_plots.png"))

plt.show()'''

'''# PPO sensitivity plots
logdir_ppo = "PPO_hyperparam_sweep_results/PPO_combined_hyperparam_results.csv"
df_ppo = pd.read_csv(logdir_ppo)

plt.style.use("ggplot")

# Aggregate RMSE by each hyperparameter
df_chunk   = df_ppo.groupby("chunk_size")["RMSE_speed"].mean().reset_index()
df_batch   = df_ppo.groupby("batch_size")["RMSE_speed"].mean().reset_index()
df_gamma   = df_ppo.groupby("gamma")["RMSE_speed"].mean().reset_index()
df_ent     = df_ppo.groupby("ent_coeff")["RMSE_speed"].mean().reset_index()  # change to 'ent' if your column is named that
df_timesteps = df_ppo.groupby("total_timesteps")["RMSE_speed"].mean().reset_index()

# Create combined figure (wide rectangle: 2 rows x 3 columns)
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

def plot_sensitivity(ax, df, x, title):
    sns.lineplot(data=df, x=x, y="RMSE_speed", marker="o", ci=None, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x)
    ax.set_ylabel("RMSE_speed")
    ax.grid(True)

axes_flat = axes.ravel()

# Row 1
plot_sensitivity(axes_flat[0], df_chunk,     "chunk_size",      "PPO: RMSE vs chunk_size")
plot_sensitivity(axes_flat[1], df_batch,     "batch_size",      "PPO: RMSE vs batch_size")

plot_sensitivity(axes_flat[2], df_gamma,     "gamma",           "PPO: RMSE vs gamma")
plot_sensitivity(axes_flat[3], df_ent,       "ent_coeff",       "PPO: RMSE vs ent_coeff")

plot_sensitivity(axes_flat[4], df_timesteps, "total_timesteps", "PPO: RMSE vs total_timesteps")

# Turn off the unused last subplot (2x3 = 6 slots, we only use 5)
axes_flat[5].axis("off")

plt.tight_layout()

# Save in the same folder as the CSV
save_dir = os.path.dirname(logdir_ppo)
plt.savefig(os.path.join(save_dir, "PPO_sensitivity_plots.png"))

plt.show()'''

'''
# TD3 sensitivity plots
logdir_td3 = "TD3_hyperparam_sweep_results/TD3_combined_hyperparam_results.csv"
df_td3 = pd.read_csv(logdir_td3)

plt.style.use("ggplot")

# Aggregate RMSE by each hyperparameter
df_chunk = df_td3.groupby("chunk_size")["RMSE_speed"].mean().reset_index()
df_batch = df_td3.groupby("batch_size")["RMSE_speed"].mean().reset_index()
df_gamma = df_td3.groupby("gamma")["RMSE_speed"].mean().reset_index()
df_tau   = df_td3.groupby("tau")["RMSE_speed"].mean().reset_index()

# Create combined figure (wide rectangle)
fig, axes = plt.subplots(2, 2, figsize=(14, 6))

def plot_sensitivity(ax, df, x, title):
    sns.lineplot(data=df, x=x, y="RMSE_speed", marker="o", ci=None, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x)
    ax.set_ylabel("RMSE_speed")
    ax.grid(True)

# Row 1
plot_sensitivity(axes[0, 0], df_chunk, "chunk_size", "TD3: RMSE vs chunk_size")
plot_sensitivity(axes[0, 1], df_batch, "batch_size", "TD3: RMSE vs batch_size")

# Row 2
plot_sensitivity(axes[1, 0], df_gamma, "gamma", "TD3: RMSE vs gamma")
plot_sensitivity(axes[1, 1], df_tau,   "tau",   "TD3: RMSE vs tau")

plt.tight_layout()

# Save in the same folder as the CSV
save_dir = os.path.dirname(logdir_td3)
plt.savefig(os.path.join(save_dir, "TD3_sensitivity_plots.png"))

plt.show()
'''

'''# DDPG sensitivity plots
file_ddpg = "DDPG_hyperparam_sweep_results\\DDPG_combined_hyperparam_results.csv"
save_dir = os.path.dirname(file_ddpg)
df_ddpg = pd.read_csv(file_ddpg)

plt.style.use("ggplot")

# Aggregate RMSE by each hyperparameter
df_chunk = df_ddpg.groupby("chunk_size")["RMSE_speed"].mean().reset_index()
df_batch = df_ddpg.groupby("batch_size")["RMSE_speed"].mean().reset_index()
df_gamma = df_ddpg.groupby("gamma")["RMSE_speed"].mean().reset_index()
df_tau   = df_ddpg.groupby("tau")["RMSE_speed"].mean().reset_index()

# Create combined figure
fig, axes = plt.subplots(2, 2, figsize=(14, 6))

def plot_sensitivity(ax, df, x, title):
    sns.lineplot(data=df, x=x, y="RMSE_speed", marker="o", ci=None, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x)
    ax.set_ylabel("RMSE_speed")
    ax.grid(True)

# Row 1
plot_sensitivity(axes[0, 0], df_chunk, "chunk_size", "DDPG: RMSE vs chunk_size")
plot_sensitivity(axes[0, 1], df_batch, "batch_size", "DDPG: RMSE vs batch_size")

# Row 2
plot_sensitivity(axes[1, 0], df_gamma, "gamma", "DDPG: RMSE vs gamma")
plot_sensitivity(axes[1, 1], df_tau,   "tau",   "DDPG: RMSE vs tau")

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(save_dir, "DDPG_sensitivity_plots.png"))'''