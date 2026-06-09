"""
MaskablePPO Wariant B (agresywny)
"""
import os
import random
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from sb3_wrapper import UnoSingleAgentEnv

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = []
        self.current_reward = 0.0
        self.current_length = 0

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.current_reward += rewards[0]
        self.current_length += 1

        if dones[0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            info = self.locals["infos"][0]
            self.wins.append(1 if info.get("winner", False) else 0)
            self.current_reward = 0.0
            self.current_length = 0

        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 10:
            recent_r = self.episode_rewards[-100:]
            recent_w = self.wins[-100:]
            recent_l = self.episode_lengths[-100:]
            self.logger.record("rollout/ep_rew_mean", float(np.mean(recent_r)))
            self.logger.record("rollout/win_rate", float(np.mean(recent_w)))
            self.logger.record("rollout/ep_len_mean", float(np.mean(recent_l)))
            self.logger.record("rollout/ep_count", len(self.episode_rewards))
            self.logger.dump(self.num_timesteps)

        return True


TOTAL_TIMESTEPS = 500_000
RUN_NAME = "ppo_aggressive_500"

print("Tworzenie środowiska...")
env = UnoSingleAgentEnv(num_players=2, opponent_policy="random")
env = Monitor(env, filename=None)

os.makedirs(f"logs/{RUN_NAME}", exist_ok=True)
os.makedirs(f"models/{RUN_NAME}", exist_ok=True)

print("Tworzenie modelu MaskablePPO (Wariant B - agresywny)...")
model = MaskablePPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    n_steps=512,
    batch_size=128,
    n_epochs=4,
    gamma=0.95,
    gae_lambda=0.9,
    clip_range=0.3,
    ent_coef=0.03,
    vf_coef=0.5,
    max_grad_norm=0.5,


    policy_kwargs=dict(net_arch=[128, 128]),
    tensorboard_log=f"logs/{RUN_NAME}",
    verbose=1,
)

callbacks = [
    EpisodeStatsCallback(log_freq=2048),
    CheckpointCallback(
        save_freq=50_000,
        save_path=f"models/{RUN_NAME}/checkpoints",
        name_prefix="ppo"
    )
]

print(f"Start treningu — {TOTAL_TIMESTEPS:,} kroków")
print("Monitor: tensorboard --logdir logs/")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    tb_log_name=RUN_NAME,
    progress_bar=True,
)

model.save(f"models/{RUN_NAME}/final_model")

import pandas as pd

stats_cb = callbacks[0]
df = pd.DataFrame({
    "episode": range(len(stats_cb.episode_rewards)),
    "reward": stats_cb.episode_rewards,
    "length": stats_cb.episode_lengths,
    "win": stats_cb.wins
})
df.to_csv(f"logs/{RUN_NAME}/episode_stats.csv", index=False)
print(f"✓ CSV: logs/{RUN_NAME}/episode_stats.csv")
print(f"\n✓ Trening zakończony")
print(f"  Final win rate (last 100): {np.mean(stats_cb.wins[-100:]):.2%}")