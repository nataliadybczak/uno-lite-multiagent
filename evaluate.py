"""
Ewaluacja wszystkich modeli + mixed agents
"""
import numpy as np
import torch
from sb3_contrib import MaskablePPO

from uno_env import UnoLiteEnv, NUM_ACTIONS, DRAW_ACTION
from train_dqn import MaskedDQN


def model_action(model, model_type, obs):
    """Wybiera akcję dla modelu, biorąc pod uwagę maskę."""
    mask = obs["action_mask"]
    legal = np.where(mask == 1)[0]

    if len(legal) == 0:
        return DRAW_ACTION

    if model_type == "random":
        return int(np.random.choice(legal))

    if model_type == "ppo":
        action, _ = model.predict(
            obs["observation"],
            action_masks=mask.astype(bool),
            deterministic=True
        )
        if not mask[int(action)]:
            return int(np.random.choice(legal))
        return int(action)

    if model_type == "dqn":
        flat_obs = np.concatenate([obs["observation"], mask.astype(np.float32)])
        action, _ = model.predict(flat_obs, deterministic=True)

        if not mask[int(action)]:
            return int(np.random.choice(legal))
        return int(action)

    raise ValueError(f"Nieznany model_type: {model_type}")


def evaluate(agent_configs, n_episodes=500, num_players=2):
    """
    agent_configs: dict {agent_name: (model, model_type)}
    """
    env = UnoLiteEnv(num_players=num_players)

    wins = {agent: 0 for agent in env.possible_agents}
    rewards_sum = {agent: 0.0 for agent in env.possible_agents}
    episode_lengths = []
    truncated_count = 0

    for ep in range(n_episodes):
        env.reset(seed=ep + 10000)
        ep_rewards = {agent: 0.0 for agent in env.possible_agents}
        ep_length = 0
        winner_found = None

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            ep_rewards[agent] += reward

            if info.get("winner") is True and winner_found is None:
                winner_found = agent

            if term or trunc:
                env.step(None)
                continue

            model, model_type = agent_configs[agent]
            action = model_action(model, model_type, obs)
            env.step(action)
            ep_length += 1

        if winner_found:
            wins[winner_found] += 1
        else:
            truncated_count += 1

        for agent in env.possible_agents:
            rewards_sum[agent] += ep_rewards[agent]

        episode_lengths.append(ep_length)

    env.close()

    results = {}
    for agent in env.possible_agents:
        results[agent] = {
            "win_rate": wins[agent] / n_episodes,
            "avg_reward": rewards_sum[agent] / n_episodes,
        }
    results["avg_length"] = float(np.mean(episode_lengths))
    results["truncated_pct"] = truncated_count / n_episodes
    return results


def print_result(label, res):
    p0_wr = res["player_0"]["win_rate"]
    p1_wr = res["player_1"]["win_rate"]
    trunc = res["truncated_pct"]
    length = res["avg_length"]
    print(f"{label:<45} {p0_wr:>7.1%} {p1_wr:>7.1%} {length:>6.1f} {trunc:>7.1%}")


if __name__ == "__main__":
    print("Ładowanie modeli...")
    ppo_base = MaskablePPO.load("models/ppo_base/final_model")
    ppo_aggr = MaskablePPO.load("models/ppo_aggressive_500/final_model")
    dqn = MaskedDQN.load("models/dqn/final_model")

    print(f"\n{'=' * 85}")
    print(f"{'Konfiguracja':<45} {'P0 win%':>8} {'P1 win%':>8} {'Len':>6} {'Trunc%':>8}")
    print(f"{'=' * 85}")

    # Baseline
    print_result("Random vs Random", evaluate({
        "player_0": (None, "random"),
        "player_1": (None, "random"),
    }))

    print(f"{'-' * 85}")

    print_result("PPO Wariant A vs Random", evaluate({
        "player_0": (ppo_base, "ppo"),
        "player_1": (None, "random"),
    }))

    print_result("PPO Wariant B vs Random", evaluate({
        "player_0": (ppo_aggr, "ppo"),
        "player_1": (None, "random"),
    }))

    print_result("DQN vs Random", evaluate({
        "player_0": (dqn, "dqn"),
        "player_1": (None, "random"),
    }))

    print(f"{'-' * 85}")
    print("MIXED — różne algorytmy w jednym epizodzie (wymaganie 6 pkt):")

    print_result("PPO Wariant A vs PPO Wariant B", evaluate({
        "player_0": (ppo_base, "ppo"),
        "player_1": (ppo_aggr, "ppo"),
    }))

    print_result("PPO Wariant A vs DQN", evaluate({
        "player_0": (ppo_base, "ppo"),
        "player_1": (dqn, "dqn"),
    }))

    print(f"{'-' * 85}")
    print("Sanity check (self-play - powinno być ~50/50):")

    print_result("PPO Wariant A vs PPO Wariant A", evaluate({
        "player_0": (ppo_base, "ppo"),
        "player_1": (ppo_base, "ppo"),
    }))

    print_result("DQN vs DQN", evaluate({
        "player_0": (dqn, "dqn"),
        "player_1": (dqn, "dqn"),
    }))

    print(f"{'=' * 85}")