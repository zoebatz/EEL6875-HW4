import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.noise import NormalActionNoise

import torch
import torch.nn as nn
import os
import time
import argparse
import csv

DATA_LEN = 1200
CSV_FILE = "speed_profile.csv"
SEED = 27

MODEL_DICT = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3, 'DDPG': DDPG}
REWARD_MODE = 'huber' # squared or huber
HUBER_DELTA = 1.0   # huber threshold


# ------------------------------------------------------------------------
# 2) Utility: chunk the dataset, possibly with leftover
# ------------------------------------------------------------------------
def chunk_into_episodes(data, chunk_size):
    """
    Splits `data` into chunks of length `chunk_size`.
    If leftover < chunk_size remains, it becomes a smaller final chunk.
    """
    episodes = []
    start = 0
    while start < len(data):
        end = start + chunk_size
        chunk = data[start:end]
        episodes.append(chunk)
        start = end
    return episodes

# ------------------------------------------------------------------------
# 3A) Training Environment: picks a random chunk each reset
# ------------------------------------------------------------------------
class TrainEnv(gym.Env):
    """
    Speed-following training environment:
      - The dataset is split into episodes of length `chunk_size`.
      - Each reset(), we pick one chunk at random.
      - lead car speed determined by dataset.
      - ego car (agent) must follow lead car's speed and maintain 5 - 30m distance.
      - action: acceleration in [-2,2]
      - observation: [v_ego, v_lead, d, v_rel, a_prev]
        - v_ego: agent's current speed
        - v_lead: lead's current speed (from dataset)
        - d: distance (x_lead - x_ego)
        - v_rel: relative speed (v_ego - v_lead)
        - a_prev: previous acceleration (minimize jerk)
      - reward: 
    """

    def __init__(self, episodes_list, delta_t=1.0, 
                 d_init_range=(15.0, 25.0), # helps at reset to minimize penalities
                 d_min=5.0, d_max=30.0, # min and max follow distance
                 lambda_d=1.0, # reward weight, follow distance most important
                 lambda_v=0.5, # reward weight, speed secondary to distance
                 lambda_j=0.1, # reward weight, minimize jerk for comfort but not a safety concern
                 ):
        super().__init__()
        self.episodes_list = episodes_list
        self.num_episodes = len(episodes_list)
        self.delta_t = delta_t
        self.d_min, self.d_max = d_min, d_max
        self.lambda_d, self.lambda_v, self.lambda_j = lambda_d, lambda_v, lambda_j

        # Actions
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

        # Observations   
        low = np.array([0.0, 0.0, 0.0, -50.0, -2.0], dtype=np.float32)    
        high = np.array([50.0, 50.0, 100.0, 50.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(5,), dtype=np.float32)

        # Episode-specific
        self.current_episode = None
        self.episode_len = 0
        self.step_idx = 0

        # Kinematics for lead and ego
        self.x_ego = self.v_ego = self.a_prev = 0.0
        self.x_lead = self.v_lead = 0.0
        self.d_init_range = d_init_range # randomize inital gap each reset


    # helper function for observations
    def _make_obs(self):
        d = self.x_lead - self.x_ego
        v_rel = self.v_ego - self.v_lead
        return np.array([self.v_ego, self.v_lead, d, v_rel, self.a_prev], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick random chunk from episodes_list
        ep_idx = np.random.randint(0, self.num_episodes)
        self.current_episode = self.episodes_list[ep_idx]
        self.episode_len = len(self.current_episode)
        self.step_idx = 0

        # Initialize kinematics
        self.v_ego = 0.0
        self.a_prev = 0.0
        self.v_lead = float(self.current_episode[self.step_idx]) # lead speed first element of chunk
        self.x_ego = 0.0

        # start lead ahead randomly (between 15-25m), realistic and easier for learning
        self.x_lead = np.random.uniform(*self.d_init_range)

        obs = self._make_obs()   # helper funct
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -2.0, 2.0)
        dt = self.delta_t

        # update ego
        self.v_ego = max(0.0, self.v_ego + accel * dt)  # above 0.0
        self.x_ego = self.x_ego + self.v_ego * dt

        # update lead
        self.v_lead = float(self.current_episode[self.step_idx])
        self.x_lead = self.x_lead + self.v_lead * dt

        # distance and jerk
        d = self.x_lead - self.x_ego
        j = (accel - self.a_prev) / dt

        # reward
        speed_diff =  abs(self.v_ego - self.v_lead)

        if d < self.d_min:
            dist_penalty = (self.d_min - d) # too close
        elif d > self.d_max:
            dist_penalty = (d - self.d_max) # too far
        else:
            dist_penalty = 0.0  # acceptable distance

        reward = (-self.lambda_d * dist_penalty
                  - self.lambda_v * speed_diff
                  - self.lambda_j * (j ** 2))
        
        # update prev accel
        self.a_prev = accel

        self.step_idx += 1
        terminated = (self.step_idx >= self.episode_len)
        truncated = False

        obs = self._make_obs()
        info = {"speed_diff": speed_diff,
                "distance": d,
                "jerk": j}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    Speed-following testing environment:
      - We run through the entire 1200-step dataset in one go.
      - observation: [current_speed, reference_speed]
      - reward: -|current_speed - reference_speed|
    """

    def __init__(self, full_data, delta_t=1.0,
                 d_init=20.0, d_min=5.0, d_max=30.0):
        super().__init__()
        self.full_data = full_data
        self.n_steps = len(full_data)
        self.delta_t = delta_t
        self.d_min, self.d_max = d_min, d_max

        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0, -50.0, -2.0], dtype=np.float32)    
        high = np.array([50.0, 50.0, 100.0, 50.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(5,), dtype=np.float32)

        self.idx = 0
        self.x_ego = self.v_ego = self.a_prev = 0.0
        self.x_lead = d_init
        self.v_lead = 0.0

    # helper function for observations
    def _make_obs(self):
        d = self.x_lead - self.x_ego
        v_rel = self.v_ego - self.v_lead
        return np.array([self.v_ego, self.v_lead, d, v_rel, self.a_prev], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.x_ego = 0.0
        self.v_ego = 0.0
        self.a_prev = 0.0
        self.x_lead = 20.0
        self.v_lead = float(self.current_episode[self.step_idx])

        obs = self._make_obs()
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -2.0, 2.0)
        dt = self.delta_t

        self.v_ego = max(0.0, self.v_ego + accel * dt) 
        self.x_ego = self.x_ego + self.v_ego * dt

        self.v_lead = float(self.full_data[self.idx])
        self.x_lead = self.x_lead + self.v_lead * dt

        # distance and jerk
        d = self.x_lead - self.x_ego
        j = (accel - self.a_prev) / dt

        speed_diff =  abs(self.v_ego - self.v_lead)

        # compute reward for analysis
        if d < self.d_min:
            dist_penalty = (self.d_min - d) # too close
        elif d > self.d_max:
            dist_penalty = (d - self.d_max) # too far
        else:
            dist_penalty = 0.0  # acceptable distance

        reward = (-1.0 * dist_penalty
                  - 0.5 * speed_diff
                  - 0.1 * (j ** 2))
        
        self.a_prev = accel
        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = self._make_obs()
        info = {"speed_diff": speed_diff,
                "distance": d,
                "jerk": j}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 4) CustomLoggingCallback (optional)
# ------------------------------------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir, log_name="training_log.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, log_name)
        self.episode_rewards = []
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'average_reward'])

    def _on_step(self):
        t = self.num_timesteps
        reward = self.locals.get('rewards', [0])[-1]
        self.episode_rewards.append(reward)

        if self.locals.get('dones', [False])[-1]:
            avg_reward = np.mean(self.episode_rewards)
            with open(self.log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward])
            self.logger.record("reward/average_reward", avg_reward)
            self.episode_rewards.clear()

        return True


# ------------------------------------------------------------------------
# 5) Main: user sets chunk_size from command line, train, then test
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs_chunk_training",
        help="Directory to store logs and trained model."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Episode length for training (e.g. 50, 100, 200)."
    )
    args = parser.parse_args()

    # --- set model ---
    model_name = 'SAC' # (SAC, PPO, TD3, DDPG)

    # ---- set random seed for reproducability ----
#    SEED = 27
    print(f"[INFO] Using random seed: {SEED}")
    np.random.seed(SEED)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    
    # ------------------------------------------------------------------------
    # 1) Always create a 1200-step speed dataset
    # ------------------------------------------------------------------------
#    DATA_LEN = 1200
#    CSV_FILE = "speed_profile.csv"

    if not os.path.exists(CSV_FILE):
        # Force-generate a 1200-step sinusoidal + noise speed profile
        speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * np.random.randn(DATA_LEN)
        df_fake = pd.DataFrame({"speed": speeds})
        df_fake.to_csv(CSV_FILE, index=False)
        print(f"Created {CSV_FILE} with {DATA_LEN} steps.")
    else:
        print(f"[INFO]: Using existing dataset: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)
    full_speed_data = df["speed"].values
    assert len(full_speed_data) == DATA_LEN, "Dataset must be 1200 steps after generation."


    chunk_size = args.chunk_size
    print(f"[INFO] Using chunk_size = {chunk_size}")

    # ---- read hyperparams from environment variables if set ----
    lr_env = os.getenv("LR")
    batch_env = os.getenv("BATCH_SIZE")
    buffer_env = os.getenv("BUFFER_SIZE")
    steps_env = os.getenv("TOTAL_STEPS")

    lr_value = float(lr_env) if lr_env else 3e-4
    batch_value = int(batch_env) if batch_env else 256
    buffer_value = int(buffer_env) if buffer_env else 500_000
    total_timesteps = int(steps_env) if steps_env else 100_000

    print(f"[INFO] Using hyperparams: LR={lr_value}, Batch={batch_value}, Buffer={buffer_value}, Steps={total_timesteps}, Reward={REWARD_MODE}")


    # ---- save each run to timestamped folder ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"{args.output_dir}_{model_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])
    print(f"[INFO] Log directory created: {log_dir}")

    log_name = f"training_log_{timestamp}.csv"

    

    # 5A) Split the 1200-step dataset into chunk_size episodes
    episodes_list = chunk_into_episodes(full_speed_data, chunk_size)
    print(f"Number of episodes: {len(episodes_list)} (some leftover if 1200 not divisible by {chunk_size})")

    # 5B) Create the TRAIN environment
    def make_train_env():
        return TrainEnv(episodes_list, delta_t=1.0,
                        d_init_range=(15.0, 25.0), # helps at reset to minimize penalities
                        d_min=5.0, d_max=30.0, # min and max follow distance
                        lambda_d=1.0, # reward weight, follow distance most important
                        lambda_v=0.5, # reward weight, speed secondary to distance
                        lambda_j=0.1, # reward weight, minimize jerk for comfort but not a safety concern
                        )
    
    num_envs = 8
    train_env = SubprocVecEnv([make_train_env for _ in range(num_envs)])

 #   train_env = DummyVecEnv([make_train_env])
    train_env.seed(SEED)    # set environment seed

    # 5C) Build the model (SAC with MlpPolicy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")


    # ---- for TD3 and DDPG ---- 
    # needs noise for exploration
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)

    # come back to clean with MODEL_DICT
    '''    
    MODEL_CONFIG = {
        'SAC': {'tau': 0.005, }
    }

    common_kwargs = dict(
        seed=SEED,
        learning_rate=lr_value,
        batch_size=batch_value,

    )
    
# Choose model dynamically
model_name = 'TD3'  # can loop through this
ModelClass = MODEL_MAP[model_name]

# Create model
model = ModelClass('MlpPolicy', train_env, **common_kwargs, **MODEL_CONFIG[model_name])
'''

    # try SAC, PPO, TD3, DDPG
    '''model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate= lr_value, # 3e-4,
        buffer_size=buffer_value,
        batch_size=batch_value,
        action_noise=action_noise,
#        learning_starts=10_000,
        tau=0.005,
        gamma=0.99,
        train_freq=100, # how often to update model
        gradient_steps=100, # how many updates per training step
        device=device
    )'''
    '''model = TD3(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate= lr_value, # 3e-4,
        buffer_size=buffer_value,
        batch_size=batch_value,
        action_noise=action_noise,
        tau=0.005,
        gamma=0.99,
        train_freq=100, # how often to update model
        gradient_steps=100, # how many updates per training step
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        device=device
    )'''
    '''model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate= lr_value, # 3e-4,
        batch_size=batch_value,
        n_steps=2048,
        n_epochs=10,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.005,
        device=device
    )'''
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate= lr_value, # 3e-4,
        batch_size=batch_value,
        buffer_size=buffer_value,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        device=device
    )
    

    model.set_logger(logger)

  #  total_timesteps = 500_000 defined above now
    callback = CustomLoggingCallback(log_dir, log_name)

    print(f"[INFO] Start training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,   # originally 100
        callback=callback
    )
    end_time = time.time()
    print(f"[INFO] Training finished in {end_time - start_time:.2f}s")

    # 5D) Save the model
    save_path = os.path.join(log_dir, f"{model_name}_speed_follow")
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}.zip")

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go (ACC)
    # ------------------------------------------------------------------------
    test_env = TestEnv(
        full_data=full_speed_data,
        delta_t=0.1,   # match training dt
        d_init=20.0,   # start mid-band
        d_min=5.0, d_max=30.0
    )

    obs, _ = test_env.reset()
    v_ego_list, v_lead_list, d_list, j_list = [], [], [], []
    rewards, actions = [], []

    for _ in range(DATA_LEN):
        action, _ = model.predict(obs, deterministic=True)  # no exploration in testing
        obs, reward, terminated, truncated, info = test_env.step(action)

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

    # ---- Convert to arrays
    v_ego_arr = np.array(v_ego_list)
    v_lead_arr = np.array(v_lead_list)
    d_arr = np.array(d_list)
    j_arr = np.array(j_list)

    # ---- Metrics required for report
    speed_diff = np.abs(v_ego_arr - v_lead_arr)
    mae_speed = float(np.mean(speed_diff))
    rmse_speed = float(np.sqrt(np.mean((v_ego_arr - v_lead_arr) ** 2)))
    jerk_mean = float(np.mean(j_arr))
    jerk_var = float(np.var(j_arr))
    in_range_pct = float(np.mean((d_arr >= 5.0) & (d_arr <= 30.0)) * 100.0)
    avg_test_reward = float(np.mean(rewards))
    corr = float(np.corrcoef(v_ego_arr, v_lead_arr)[0, 1])

    print(f"[TEST] MAE={mae_speed:.3f}, RMSE={rmse_speed:.3f}, "
        f"Jerk μ={jerk_mean:.3f}, Jerk σ²={jerk_var:.3f}, "
        f"InRange%={in_range_pct:.1f}, Corr={corr:.3f}")

    # ---- Save metrics
    metrics_path = os.path.join(log_dir, f"acc_test_metrics_chunk{chunk_size}_{timestamp}.csv")
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["avg_test_reward", "MAE_speed", "RMSE_speed",
                        "jerk_mean", "jerk_var", "in_range_pct",
                        "Training Time", "Corr Coeff"])
        writer.writerow([avg_test_reward, mae_speed, rmse_speed,
                        jerk_mean, jerk_var, in_range_pct,
                        round(end_time - start_time, 2), corr])

    # ---- Plots required for the report
    # 1) Ego vs Lead speed
    plt.figure(figsize=(10, 5))
    plt.plot(v_lead_arr, label="Lead Speed", linestyle="--")
    plt.plot(v_ego_arr, label="Ego Speed", linestyle="-")
    plt.xlabel("Timestep"); plt.ylabel("Speed (m/s)")
    plt.title(f"ACC: Ego vs Lead Speeds (chunk_size={chunk_size})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"acc_speed_tracking_chunk{chunk_size}.png"))

    # 2) Distance over time with safe band [5,30] m
    plt.figure(figsize=(10, 4))
    plt.plot(d_arr, label="Following Distance")
    plt.axhspan(5.0, 30.0, alpha=0.15, label="Safe Range [5,30] m")
    plt.xlabel("Timestep"); plt.ylabel("Distance (m)")
    plt.title("Following Distance Over Time")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"acc_distance_over_time_chunk{chunk_size}.png"))

    # 3) Jerk over time
    plt.figure(figsize=(10, 4))
    plt.plot(j_arr)
    plt.xlabel("Timestep"); plt.ylabel("Jerk (m/s³)")
    plt.title(f"Jerk Over Time (mean={jerk_mean:.3f}, var={jerk_var:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"acc_jerk_over_time_chunk{chunk_size}.png"))

    # 4) Speed difference |v_ego - v_lead|
    plt.figure(figsize=(10, 4))
    plt.plot(speed_diff)
    plt.xlabel("Timestep"); plt.ylabel("|Δv| (m/s)")
    plt.title(f"Speed Difference (MAE={mae_speed:.3f}, RMSE={rmse_speed:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"acc_speed_diff_chunk{chunk_size}.png"))


'''
    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    test_env = TestEnv(full_speed_data, delta_t=1.0)

    obs, _ = test_env.reset()
    predicted_speeds = []
    reference_speeds = []
    rewards = []

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        predicted_speeds.append(obs[0])  # current_speed
        reference_speeds.append(obs[1])  # reference_speed
        rewards.append(reward)
        if terminated or truncated:
            break

    # ---- MAE and RMSE ----
    predicted_speeds = np.array(predicted_speeds)
    reference_speeds = np.array(reference_speeds)

    mae = np.mean(np.abs(predicted_speeds - reference_speeds))
    rmse = np.sqrt(np.mean((predicted_speeds - reference_speeds)**2))
    print(f"[TEST] MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    avg_test_reward = np.mean(rewards)
    print(f"[TEST] Average reward over 1200-step test: {avg_test_reward:.3f}")

    # ---- correlation coefficient ----
    corr = np.corrcoef(predicted_speeds, reference_speeds)[0, 1]


    # ---- save metrics ----
    metrics_path = os.path.join(log_dir, f"test_metrics_chunk{chunk_size}_{timestamp}.csv")
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["avg_test_reward", "MAE", "RMSE", "Training Time", "Corr Coeff"])
        writer.writerow([avg_test_reward, mae, rmse, round(end_time - start_time, 2), corr])

    # Plot the entire test
    plt.figure(figsize=(10, 5))
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Predicted Speed", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Test on full 1200-step dataset (chunk_size={chunk_size})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"speed_tracking_chunk{chunk_size}.png"))
#     plt.show()

    # ---- plot error ----
    plt.figure(figsize=(10, 4))
    error = np.abs(predicted_speeds - reference_speeds)
    plt.plot(error, color='red', alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Absolute Error (m/s)")
    plt.title(f"Tracking Error MAE={mae:.3f}, RMSE={rmse:.3f} (chunk_size={chunk_size})")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"error_plot_chunk{chunk_size}.png"))
#    plt.show()

'''

if __name__ == "__main__":
    main()
