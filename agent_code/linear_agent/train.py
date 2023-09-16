from collections import namedtuple, deque

import numpy as np
import os
import pickle
from typing import List
from .constants import *

from .callbacks import state_to_features, get_bomb_explosion_fields, contains_crate

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


def update_model(self, y, old_features, self_action):
    action_index = ACTIONS.index(self_action)
    self.model[:, action_index] += self.config['learning_rate'] * (old_features * (y - np.dot(self.model[:, action_index], old_features)))

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
    action_index = ACTIONS.index(action)
    state_vector = state_to_features(old_game_state)

    coin_pos_features = state_vector[COIN_POS_FEATURES_START : COIN_POS_FEATURES_END]
    coin_dist_features = state_vector[COIN_DIST_FEATURES_START : COIN_DIST_FEATURES_END]

    crate_pos_features = state_vector[CRATE_POS_FEATURES_START : CRATE_POS_FEATURES_END]
    crate_dist_features = state_vector[CRATE_DIST_FEATURES_START : CRATE_DIST_FEATURES_END]

    live_saving_features = state_vector[LIVE_SAVING_FEATURES_START : LIVE_SAVING_FEATURES_END]
    deadly_features = state_vector[DEADLY_FEATURES_START : DEADLY_FEATURES_END]
    bomb_survivable_feature = state_vector[BOMB_SURVIVABLE_FEATURES_START : BOMB_SURVIVABLE_FEATURES_END]
    bomb_usefull_feature = state_vector[BOMB_USEFULL_FEATURES_START : BOMB_USEFULL_FEATURES_END]

    def feature_action_picked(feature_vector):
        return action_index < len(feature_vector) and feature_vector[action_index] != 0

    if np.any(deadly_features): 
        if feature_action_picked(deadly_features):
            events.append(DEADLY_MOVE_CHOOSEN)
            return
        else:
            events.append(DEADLY_MOVE_AVOIDED)
    
    if np.any(live_saving_features != 0):
        if feature_action_picked(live_saving_features):
            events.append(LIVE_SAVING_MOVE_CHOOSEN)
        else:
            events.append(LIVE_SAVING_MOVE_AVOIDED)
        return
    
    if bomb_survivable_feature[0] == 0:
        if action == 'BOMB':
            events.append(UNSURVIVABLE_BOMB_CHOOSEN)
            return
        else:
            events.append(SURVIVABLE_BOMB_CHOOSEN)
    
    if action == 'BOMB':
        #explosion_fields = get_bomb_explosion_fields(old_game_state['self'][3], old_game_state['field'])
        #if contains_crate(explosion_fields, old_game_state['field']):
        #    events.append(USEFULL_BOMB)
        #else:
        #    events.append(USELESS_BOMB)
        
        if bomb_usefull_feature[0] != 0:
            events.append(USEFULL_BOMB)
        else:
            events.append(USELESS_BOMB)
    
    if np.any(coin_pos_features != 0):
        if feature_action_picked(coin_pos_features):
            events.append(COIN_MOVE_CHOOSEN)
        else:
            events.append(COIN_MOVE_AVOIDED)
        return
    
    if np.any(crate_pos_features != 0):
        if feature_action_picked(crate_pos_features):
            events.append(CRATE_MOVE_CHOOSEN)
        else:
            events.append(CRATE_MOVE_AVOIDED)
        return

    #if e.WAITED in events and np.all(old_game_state['explosion_map'] == 0):
    #    events.append(UNNECESSARY_WAITING)    


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
    
    if len(self.rewards) >= self.config['learning_stepsize']:
        y = self.q_values[-1]
        for j in range(-1, -self.config['learning_stepsize'] - 1, -1):
            y = self.rewards[j] + self.config['gamma'] * y
        update_model(self, y, old_features, self_action)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    old_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(old_features, last_action, None, reward))

    self.features.append(old_features)
    self.actions.append(last_action)
    self.rewards.append(reward)

    for i in range(len(self.rewards) - self.config['learning_stepsize'], len(self.rewards)):
        y = 0
        for j in reversed(range(i, len(self.rewards))):
            y = self.rewards[j] + self.config['gamma'] * y
        update_model(self, y, self.features[i], self.actions[i])

    reset_lists(self)

    with open(self.config['model_filename'], "wb") as file:
        pickle.dump(self.model, file)

    self.round_counter += 1


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = self.config['rewards']
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
