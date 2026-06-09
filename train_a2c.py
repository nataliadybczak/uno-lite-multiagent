"""
A2C - drugi algorytm do porównania z MaskablePPO.
A2C jest policy gradient (jak PPO), ale prostszy:
- bez clip range
- synchroniczny actor-critic
- brak natywnego masking, ale radzi sobie z karą za nielegalne akcje
"""
import os
import random
import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces

from sb3_wrapper import UnoSingleAgentEnv
from uno_env import NUM_ACTIONS

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


# ── Wrapper: dokleja maskę do obs + kara za nielegalne ─────────────
class A2CCompatibleWrapper(gym.Wrapper):
    """
    Obs zawiera oryginalną obserwację + maskę (ostatnie 65 bitów).
    Nielegalna akcja → silna kara (-1.0) + wymuszenie losowej legalnej.
    """
    def __init__(self, env):
        super().__init__(env)
        orig_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=0, high=20,
            shape=(orig_dim + NUM_ACTIONS,),
            dtype=np.float32
        )
        self.action_space = env.action_space
        self._last_mask = None

    def _build_obs(self, obs, mask):
        return np.concatenate([obs, mask.astype(np.float32)]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_mask = self.env.action_masks()
        return self._build_obs(obs, self._last_mask), info

    def step(self, action):
        illegal_penalty = 0.0
        if self._last_mask is not None and not self._last_mask[action]:
            illegal_penalty = -0.1
            legal = np.where(self._last_mask)[0]
            if len(legal) > 0:
                action = int(np.random.choice(legal))

        obs, reward, term, trunc, info = self.env.step(action)
        reward += illegal_penalty
        self._last_mask = self.env.action_masks()
        return self._build_obs(obs, self._last_mask), reward, term, trunc, info


# ── Callback ────────────────────────────────────────────────────────
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


TOTAL_TIMESTEPS = 200_000
RUN_NAME = "a2c"

print("Tworzenie środowiska...")
base_env = UnoSingleAgentEnv(num_players=2, opponent_policy="random")
env = A2CCompatibleWrapper(base_env)
env = Monitor(env, filename=None)

os.makedirs(f"logs/{RUN_NAME}", exist_ok=True)
os.makedirs(f"models/{RUN_NAME}", exist_ok=True)

print("Tworzenie modelu A2C...")
model = A2C(
    policy="MlpPolicy",
    env=env,

    learning_rate=1e-4,
    n_steps=128,                    # A2C działa z bardzo krótkimi rolloutami
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_rms_prop=True,            # standard dla A2C

    policy_kwargs=dict(net_arch=[128, 128]),
    tensorboard_log=f"logs/{RUN_NAME}",
    verbose=1,
)

callbacks = [
    EpisodeStatsCallback(log_freq=2048),
    CheckpointCallback(
        save_freq=50_000,
        save_path=f"models/{RUN_NAME}/checkpoints",
        name_prefix="a2c"
    )
]

print(f"Start treningu — {TOTAL_TIMESTEPS:,} kroków")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    tb_log_name=RUN_NAME,
    progress_bar=True,
)

model.save(f"models/{RUN_NAME}/final_model")

import pandas as pd
stats_cb = callbacks[0]
pd.DataFrame({
    "episode": range(len(stats_cb.episode_rewards)),
    "reward": stats_cb.episode_rewards,
    "length": stats_cb.episode_lengths,
    "win": stats_cb.wins
}).to_csv(f"logs/{RUN_NAME}/episode_stats.csv", index=False)

print(f"\n✓ Trening zakończony")
print(f"  Final win rate (last 100): {np.mean(stats_cb.wins[-100:]):.2%}")