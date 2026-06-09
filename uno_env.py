"""
UNO Lite

Talia (uproszczona, 100 kart):
- 4 kolory: red, blue, green, yellow
- dla każdego koloru:
    - 1 karta o wartości 0
    - 2 karty każdej wartości od 1 do 9
    - 2 karty akcji Skip
    - 2 karty akcji Reverse
    - 2 karty akcji Draw Two

Akcje (przestrzeń Discrete(101)):
- 0-99: zagranie konkretnej karty
- 100: dobranie karty z talii

Obserwacja:
- hand_vec (100 elementów)
    liczba posiadanych kart każdego typu
- top_vec (100 elementów one-hot)
    karta znajdująca się na szczycie stosu
- counts (n_players elementów)
    liczba kart każdego gracza (znormalizowana)
- direction (1 element)
    kierunek gry (+1 lub -1)

Dla dwóch graczy:

100 + 100 + 2 + 1 = 203 elementy

Action mask:
- 101 elementów
- 1 oznacza akcję legalną
- dobranie karty jest zawsze legalne
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AgentID, ObsType

# Definicja talii
COLORS = ["red", "blue", "green", "yellow"]
NUMBERS = list(range(10))  # 0-9
ACTIONS = ["skip", "reverse", "draw2"]


def build_deck_definition():
    """Zwraca listę 64 definicji kart (color, value)"""
    deck = []
    for color in COLORS:
        # 1x karta "0"
        deck.append((color, "0"))
        # 2x karty 1-9
        for n in range(1, 10):
            deck.append((color, str(n)))
            deck.append((color, str(n)))
        # 2x każda akcja
        for a in ACTIONS:
            deck.append((color, a))
            deck.append((color, a))
    return deck


CARD_DEFINITIONS = build_deck_definition()
NUM_CARDS = len(CARD_DEFINITIONS)
DRAW_ACTION = NUM_CARDS
NUM_ACTIONS = NUM_CARDS + 1


def cards_match(card_a, card_b):
    color_a, val_a = card_a
    color_b, val_b = card_b
    return color_a == color_b or val_a == val_b


# Środowisko
class UnoLiteEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "uno_lite_v0",
        "is_parallelizable": False,
    }

    def __init__(self, num_players=2, render_mode=None, max_steps=500):
        super().__init__()

        assert 2 <= num_players <= 4
        self.num_players = num_players
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.possible_agents = [f"player_{i}" for i in range(num_players)]

        # Obserwacja
        obs_dim = NUM_CARDS + NUM_CARDS + num_players + 1

        self._observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(
                    low=0, high=20,
                    shape=(obs_dim,),
                    dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1,
                    shape=(NUM_ACTIONS,),
                    dtype=np.int8
                )
            })
            for agent in self.possible_agents
        }

        self._action_spaces = {
            agent: spaces.Discrete(NUM_ACTIONS)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # Reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif not hasattr(self, "_np_random"):
            self._np_random = np.random.default_rng()

        self.deck = list(range(NUM_CARDS))
        self._np_random.shuffle(self.deck)

        self.hands = {agent: [] for agent in self.possible_agents}
        for _ in range(7):
            for agent in self.possible_agents:
                self.hands[agent].append(self.deck.pop())

        while True:
            top = self.deck.pop()
            color, value = CARD_DEFINITIONS[top]
            if value not in ACTIONS:
                self.discard = [top]
                break
            else:
                self.deck.insert(0, top)

        # Stan gry
        self.direction = 1
        self.step_count = 0
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def _compute_action_mask(self, agent):
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        hand = self.hands[agent]
        top_card = CARD_DEFINITIONS[self.discard[-1]]

        has_playable = False
        for card_idx in hand:
            card = CARD_DEFINITIONS[card_idx]
            if cards_match(card, top_card):
                mask[card_idx] = 1
                has_playable = True

        mask[DRAW_ACTION] = 1

        return mask

    # Obserwacja
    def observe(self, agent):
        hand = self.hands[agent]
        top_card = self.discard[-1]

        hand_vec = np.zeros(NUM_CARDS, dtype=np.float32)
        for card_idx in hand:
            hand_vec[card_idx] += 1

        top_vec = np.zeros(NUM_CARDS, dtype=np.float32)
        top_vec[top_card] = 1

        my_idx = self.possible_agents.index(agent)
        counts = []
        for i in range(self.num_players):
            rel_idx = (my_idx + i) % self.num_players
            counts.append(len(self.hands[self.possible_agents[rel_idx]]) / 20.0)
        counts = np.array(counts, dtype=np.float32)


        direction_vec = np.array([self.direction], dtype=np.float32)

        observation = np.concatenate([
            hand_vec, top_vec, counts, direction_vec
        ])

        action_mask = self._compute_action_mask(agent)
        return {"observation": observation, "action_mask": action_mask}

    def _draw_card(self, agent):
        """Dobiera 1 kartę z talii dla agenta. Przetasowuje jeśli pusta."""
        if not self.deck:
            top = self.discard.pop()
            self.deck = self.discard[:]
            self._np_random.shuffle(self.deck)
            self.discard = [top]
        if self.deck:
            self.hands[agent].append(self.deck.pop())

    # Step
    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self.step_count += 1
        self.rewards = {a: 0.0 for a in self.agents}

        if action == DRAW_ACTION:
            self._draw_card(agent)
            self.rewards[agent] = -0.25
            self._advance_turn(skip=False)

        else:
            if action not in self.hands[agent]:
                self.rewards[agent] = -1.0
                self._advance_turn(skip=False)
                self._accumulate_rewards()
                return

            card = CARD_DEFINITIONS[action]
            top_card = CARD_DEFINITIONS[self.discard[-1]]

            if not cards_match(card, top_card):
                self.rewards[agent] = -1.0
                self._advance_turn(skip=False)
                self._accumulate_rewards()
                return

            cards_before = len(self.hands[agent])
            self.hands[agent].remove(action)
            self.discard.append(action)
            cards_after = len(self.hands[agent])

            # nagroda za zmniejszenie ręki
            self.rewards[agent] += 0.02

            color, value = card

            if value == "draw2":
                self.rewards[agent] += 0.05
            elif value in ("skip", "reverse"):
                self.rewards[agent] += 0.03
            else:
                self.rewards[agent] += 0.01

            if len(self.hands[agent]) == 0:
                self.rewards[agent] += 10.0
                self.infos[agent] = {"winner": True}
                for other in self.agents:
                    if other != agent:
                        self.rewards[other] = -10.0
                        self.infos[other] = {"winner": False}
                self.terminations = {a: True for a in self.agents}
                self._accumulate_rewards()
                return

            skip_next = False
            if value == "skip":
                skip_next = True
            elif value == "reverse":
                self.direction *= -1
                if self.num_players == 2:
                    skip_next = True
            elif value == "draw2":
                next_agent = self._get_next_agent()
                for _ in range(2):
                    self._draw_card(next_agent)
                skip_next = True

            self._advance_turn(skip=skip_next)

        if self.step_count >= self.max_steps:
            self.truncations = {a: True for a in self.agents}

        self._accumulate_rewards()

    def _get_next_agent(self):
        """Zwraca następnego agenta zgodnie z kierunkiem."""
        current_idx = self.possible_agents.index(self.agent_selection)
        next_idx = (current_idx + self.direction) % self.num_players
        return self.possible_agents[next_idx]

    def _advance_turn(self, skip=False):
        current_idx = self.possible_agents.index(self.agent_selection)
        steps = 2 if skip else 1
        next_idx = (current_idx + self.direction * steps) % self.num_players
        next_agent = self.possible_agents[next_idx]
        for _ in range(self.num_players):
            self.agent_selection = self._agent_selector.next()
            if self.agent_selection == next_agent:
                break

    def render(self):
        if self.render_mode is None:
            return

        print(f"\n--- Krok {self.step_count} ---")
        print(f"Stos: {CARD_DEFINITIONS[self.discard[-1]]}")
        print(f"Kierunek: {'→' if self.direction == 1 else '←'}")
        print(f"Tura: {self.agent_selection}")
        for agent in self.possible_agents:
            hand_cards = [CARD_DEFINITIONS[c] for c in self.hands[agent]]
            print(f"  {agent}: {len(hand_cards)} kart")

    def close(self):
        pass


def env(**kwargs):
    return UnoLiteEnv(**kwargs)