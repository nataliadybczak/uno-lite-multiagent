"""
Wrapper: PettingZoo AEC → Gymnasium env dla SB3.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from uno_env import UnoLiteEnv, NUM_ACTIONS


class UnoSingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players=2, opponent_policy="random", render_mode=None):
        super().__init__()

        self.uno = UnoLiteEnv(num_players=num_players, render_mode=render_mode)
        self.trained_agent = "player_0"
        self.opponent_policy = opponent_policy

        obs_dim = self.uno.observation_space(
            self.trained_agent
        )["observation"].shape[0]

        self.observation_space = spaces.Box(
            low=0, high=20,
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._current_mask = np.ones(NUM_ACTIONS, dtype=bool)

    def action_masks(self):
        return self._current_mask.copy()

    def _get_flat_obs(self, agent):
        """Zwraca płaski wektor obs i aktualizuje _current_mask."""
        try:
            raw = self.uno.observe(agent)
            self._current_mask = raw["action_mask"].astype(bool)
            return raw["observation"].astype(np.float32)
        except Exception:
            obs_dim = self.observation_space.shape[0]
            self._current_mask = np.ones(NUM_ACTIONS, dtype=bool)
            return np.zeros(obs_dim, dtype=np.float32)


    def _opponent_action(self, agent):
        raw = self.uno.observe(agent)
        mask = raw["action_mask"]
        legal = np.where(mask == 1)[0]

        if self.opponent_policy == "heuristic":
            from uno_env import CARD_DEFINITIONS, DRAW_ACTION

            playable = [a for a in legal if a != DRAW_ACTION]
            if not playable:
                return DRAW_ACTION

            # Priorytet 1: draw2
            d2 = [a for a in playable if CARD_DEFINITIONS[a][1] == "draw2"]
            if d2: return int(np.random.choice(d2))
            # Priorytet 2: skip / reverse
            sr = [a for a in playable if CARD_DEFINITIONS[a][1] in ("skip", "reverse")]
            if sr: return int(np.random.choice(sr))
            # Priorytet 3: cokolwiek
            return int(np.random.choice(playable))

        if self.opponent_policy == "random" or not callable(self.opponent_policy):
            return int(np.random.choice(legal))
        return self.opponent_policy(raw)

    def _play_until_my_turn(self):
        """Pozwala przeciwnikom grać aż znów moja tura lub epizod skończy."""
        max_iters = 100
        iters = 0

        while iters < max_iters:
            iters += 1
            if (all(self.uno.terminations.values())
                    or all(self.uno.truncations.values())):
                return True

            current = self.uno.agent_selection

            if current == self.trained_agent:
                if (self.uno.terminations.get(self.trained_agent, False)
                        or self.uno.truncations.get(self.trained_agent, False)):
                    return True
                return False

            _, _, term, trunc, _ = self.uno.last()
            if term or trunc:
                self.uno.step(None)
            else:
                action = self._opponent_action(current)
                self.uno.step(action)

        return True

    def reset(self, seed=None, options=None):
        self.uno.reset(seed=seed)
        done = self._play_until_my_turn()

        if done:
            new_seed = (seed or 0) + 1
            return self.reset(seed=new_seed, options=options)

        obs = self._get_flat_obs(self.trained_agent)
        return obs, {}

    def step(self, action):
        # 1. Gra już skończona przed naszą akcją
        if (self.uno.terminations.get(self.trained_agent, False)
                or self.uno.truncations.get(self.trained_agent, False)):
            obs = self._get_flat_obs(self.trained_agent)
            winner_info = self.uno.infos.get(self.trained_agent, {})
            return obs, 0.0, True, False, winner_info

        if self.uno.agent_selection != self.trained_agent:
            self._play_until_my_turn()
            obs = self._get_flat_obs(self.trained_agent)
            return obs, 0.0, False, False, {}

        _, _, term, trunc, _ = self.uno.last()
        if term or trunc:
            self.uno.step(None)
            obs = self._get_flat_obs(self.trained_agent)
            winner_info = self.uno.infos.get(self.trained_agent, {})
            return obs, 0.0, True, False, winner_info

        self.uno.step(int(action))

        my_reward = float(self.uno.rewards.get(self.trained_agent, 0.0))
        my_term = self.uno.terminations.get(self.trained_agent, False)
        my_trunc = self.uno.truncations.get(self.trained_agent, False)

        if my_term or my_trunc:
            obs = self._get_flat_obs(self.trained_agent)
            winner_info = self.uno.infos.get(self.trained_agent, {})
            return obs, my_reward, my_term, my_trunc, winner_info

        self._play_until_my_turn()

        extra_reward = float(self.uno.rewards.get(self.trained_agent, 0.0))
        total_reward = my_reward + extra_reward

        my_term = self.uno.terminations.get(self.trained_agent, False)
        my_trunc = self.uno.truncations.get(self.trained_agent, False)

        obs = self._get_flat_obs(self.trained_agent)
        winner_info = self.uno.infos.get(self.trained_agent, {})

        return obs, total_reward, my_term, my_trunc, winner_info

    def render(self):
        self.uno.render()

    def close(self):
        self.uno.close()