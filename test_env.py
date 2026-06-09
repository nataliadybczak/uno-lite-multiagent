"""
Test środowiska UNO Lite - losowi agenci grają kilka epizodów.
"""
import numpy as np
from uno_env import UnoLiteEnv, CARD_DEFINITIONS


def play_random_episode(env, seed=0, verbose=False):
    env.reset(seed=seed)

    total_steps = 0
    rewards = {agent: 0.0 for agent in env.possible_agents}

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        rewards[agent] += reward

        if termination or truncation:
            env.step(None)
            continue

        # Wybierz LOSOWĄ ale LEGALNĄ akcję
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask == 1)[0]
        action = np.random.choice(legal_actions)

        if verbose:
            env.render()
            print(f"  {agent} → akcja {action} (legalne: {len(legal_actions)})")

        env.step(action)
        total_steps += 1

    return total_steps, rewards


if __name__ == "__main__":
    env = UnoLiteEnv(num_players=2)

    print("=" * 60)
    print("Test środowiska UNO Lite")
    print("=" * 60)
    print(f"Liczba kart w talii: {len(CARD_DEFINITIONS)}")
    print(f"Liczba akcji: {env.action_space('player_0').n}")
    print(f"Obs space: {env.observation_space('player_0')['observation'].shape}")
    print()

    # Test 1 epizod z verbose
    print("--- Epizod testowy z renderem ---")
    steps, rewards = play_random_episode(env, seed=42, verbose=False)
    print(f"Kroki: {steps}")
    print(f"Rewards: {rewards}")

    # Test 100 epizodów - statystyki
    print("\n--- 100 epizodów losowych ---")
    all_steps = []
    p0_wins = 0
    p1_wins = 0

    for ep in range(100):
        steps, rewards = play_random_episode(env, seed=ep)
        all_steps.append(steps)
        if rewards["player_0"] > rewards["player_1"]:
            p0_wins += 1
        elif rewards["player_1"] > rewards["player_0"]:
            p1_wins += 1

    print(f"Średnia długość epizodu: {np.mean(all_steps):.1f} kroków")
    print(f"Min/Max: {min(all_steps)} / {max(all_steps)}")
    print(f"Wygrane player_0: {p0_wins} | player_1: {p1_wins} | remisów: {100 - p0_wins - p1_wins}")
    print("\n✓ Środowisko działa")