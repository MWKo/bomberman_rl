from collections import namedtuple, deque

import numpy as np
import os
import pickle
from typing import List
from .constants import *

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
UNNECESSARY_WAITING = "UNNECESSARY_WAITING"
LONG_INVALID = "LONG_INVALID"

PUNISH_WAITING_INVALID_AFTER = 2000 # rounds
round_counter = 0

MAX_TOLERATED_WAITING = 5
waited_counter = 0

MAX_TOLERATED_INVALID = 1
invalid_counter = 0

LEARNING_STEPSIZE = 1

gamma = 0.98
learning_rate = 0.01


def update_model(self, y, old_features, self_action):
    action_index = ACTIONS.index(self_action)
    self.model[:, action_index] += learning_rate * (old_features * (y - np.dot(self.model[:, action_index], old_features)))

def reset_lists(self):
    self.actions = []
    self.rewards = []
    self.q_values = []
    self.features = []

def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    reset_lists(self)
    self.waited_counter = 0
    self.invalid_counter = 0
    self.round_counter = 0


def add_custom_events(self, old_game_state: dict, action: str, events: List[str]):
    if e.WAITED in events and np.all(old_game_state['explosion_map'] == 0):
        events.append(UNNECESSARY_WAITING)

    if e.INVALID_ACTION in events:
        self.invalid_counter += 1
        if self.invalid_counter >= MAX_TOLERATED_INVALID:
            events.append(LONG_INVALID)
    else:
        self.invalid_counter = 0
    


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    add_custom_events(self, old_game_state, self_action, events)


    # state_to_features is defined in callbacks.py
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events)
    self.transitions.append(Transition(old_features, self_action, new_features, reward))

    q_vector = self.model.T @ new_features
    self.features.append(old_features)
    self.actions.append(self_action)
    self.rewards.append(reward)
    self.q_values.append(np.max(q_vector))
    
    if len(self.rewards) >= LEARNING_STEPSIZE:
        y = self.q_values[-1]
        for j in range(-1, -LEARNING_STEPSIZE - 1, -1):
            y = self.rewards[j] + gamma * y
        update_model(self, y, old_features, self_action)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    old_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(old_features, last_action, None, reward))

    self.features.append(old_features)
    self.actions.append(last_action)
    self.rewards.append(reward)

    for i in range(len(self.rewards) - LEARNING_STEPSIZE, len(self.rewards)):
        y = 0
        for j in reversed(range(i, len(self.rewards))):
            y = self.rewards[j] + gamma * y
        update_model(self, y, self.features[i], self.actions[i])

    reset_lists(self)

    with open(MODEL_FILE_NAME, "wb") as file:
        pickle.dump(self.model, file)

    self.round_counter += 1

def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        #e.BOMB_DROPPED: 0.1,
        e.CRATE_DESTROYED: 0.3,
        e.KILLED_SELF: -3,
    }
    if self.round_counter >= PUNISH_WAITING_INVALID_AFTER:
        game_rewards[e.BOMB_DROPPED] = 0
        game_rewards[LONG_INVALID] = -1
        game_rewards[UNNECESSARY_WAITING] = -1
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
