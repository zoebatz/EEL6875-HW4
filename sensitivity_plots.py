import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import csv

# DDPG sensitivity plots
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
plt.savefig(os.path.join(save_dir, "DDPG_sensitivity_plots.png"))