import os
import random
import pickle

import numpy as np
from .constants import *
from settings import BOMB_TIMER, ROWS, COLS

# Callback functions
def setup(self):
    self.config = { **DEFAULT_CONFIG, **self.config } if self.config is not None else DEFAULT_CONFIG

    if not os.path.isfile(self.config['model_filename']) or (self.train and self.config['override_model']):
        self.logger.info(self.config['model_filename'])
        weights = np.random.rand(FEATURE_SIZE, len(ACTIONS))
        self.model = weights / np.abs(weights).sum(axis=0)
    else:
        self.logger.info("Loading model from saved state.")
        with open(self.config['model_filename'], "rb") as file:
            self.model = pickle.load(file)
        # print(self.model)

def act(self, game_state: dict) -> str:
    if self.train and random.random() < self.config['exploration']['epsilon']:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=self.config['exploration']['action_probabilities'])

    self.logger.debug("Querying model for action.")
    q_vector = self.model.T @ state_to_features(game_state)

    action = get_choosen_action(q_vector, game_state, filter_invalid = not self.train)
    #print_agent_choice(action, game_state, q_vector)
    return action



# helper functions

def is_valid_action(game_state: dict, action):
    if action in ['WAIT', 'BOMB']:
        return True
    
    sx, sy = game_state['self'][3]
    nx, ny = {
        'LEFT': (sx - 1, sy),
        'RIGHT': (sx + 1, sy),
        'UP': (sx, sy - 1),
        'DOWN': (sx, sy + 1)
    }[action]
    return game_state['field'][nx, ny] == 0


def get_choosen_action(q_vector, game_state, filter_invalid: bool = False):
    if not filter_invalid:
        return ACTIONS[np.argmax(q_vector)]
    
    valid_actions_mask = np.array([is_valid_action(game_state, action) for action in ACTIONS], dtype=np.bool_)
    argmax_valid_actions = np.argmax(q_vector[valid_actions_mask])
    valid_indices = np.arange(len(ACTIONS))[valid_actions_mask]
    return ACTIONS[valid_indices[argmax_valid_actions]]


def print_agent_choice(action, game_state, q_vector):
    print(action)
    print(state_to_features(game_state))
    print(dict(zip(ACTIONS, q_vector)))
    print()


def is_next_to(position, is_searched_position):
    x, y = position
    for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
        if nx < 0 or nx >= COLS or ny < 0 or ny >= ROWS:
            continue
        if is_searched_position((nx, ny)):
            return True
    return False

def get_bomb_explosion_fields(position, arena):
    x, y = position
    explosion_fields = [position]
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for i in range(1, 4):
            if arena[x + dx * i, y + dy * i] == -1:
                break
            explosion_fields.append((x + dx * i, y + dy * i))
    return explosion_fields

def contains_crate(positions, arena):
    for (x, y) in positions:
        if arena[x, y] == 1:
            return True
    return False

def find_closest_position(starting_position, game_state: dict, is_searched_position):
    arena = game_state['field']
    bombs = game_state['bombs']
    others = game_state['others']
    bomb_positions = list(map(lambda x: x[0], bombs))
    other_positions = list(map(lambda x: x[3], others))
    
    queue = [(starting_position, 'WAIT', 0)]
    visited = np.zeros(arena.shape, dtype=np.bool_)
    while len(queue) > 0:
        (x, y), action, dist = queue.pop(0)
        if visited[x, y]:
            continue
        visited[x, y] = True

        if is_searched_position((x, y), game_state):
            return action, dist
        
        for (nx, ny), naction in [((x - 1, y), "LEFT"), ((x + 1, y), "RIGHT"), ((x, y - 1), "UP"), ((x, y + 1), "DOWN")]:
            if nx < 0 or nx >= COLS or ny < 0 or ny >= ROWS:
                continue
            if arena[nx, ny] != 0 or visited[nx, ny] or (nx, ny) in bomb_positions or (nx, ny) in other_positions:
                continue
            queue.append(((nx, ny), naction if action == "WAIT" else action, dist + 1))
    
    return None, -1


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None
    
    _, _, _, self_position = game_state['self']
    sx, sy = self_position  
    
    features = np.zeros(FEATURE_SIZE)
    features[0] = 1 # bias

    coin_pos_features = features[COIN_POS_FEATURES_START : COIN_POS_FEATURES_END]
    coin_dist_features = features[COIN_DIST_FEATURES_START : COIN_DIST_FEATURES_END]

    crate_pos_features = features[CRATE_POS_FEATURES_START : CRATE_POS_FEATURES_END]
    crate_dist_features = features[CRATE_DIST_FEATURES_START : CRATE_DIST_FEATURES_END]

    live_saving_features = features[LIVE_SAVING_FEATURES_START : LIVE_SAVING_FEATURES_END]
    deadly_features = features[DEADLY_FEATURES_START : DEADLY_FEATURES_END]
    bomb_survivable_feature = features[BOMB_SURVIVABLE_FEATURES_START : BOMB_SURVIVABLE_FEATURES_END]
    bomb_usefull_feature = features[BOMB_USEFULL_FEATURES_START : BOMB_USEFULL_FEATURES_END]
    
    coin_action, dist = find_closest_position(self_position, game_state, lambda pos, state: pos in state['coins'])
    if coin_action is not None:
        coin_pos_features[ACTIONS.index(coin_action)] = 1
        coin_dist_features[ACTIONS.index(coin_action)] = 1 / (dist + 1)

    crate_action, dist = find_closest_position(self_position, game_state, 
        lambda pos, state: is_next_to(pos, lambda p: state['field'][p[0], p[1]] == 1)
    )
    if crate_action is not None:
        if dist > 0:
            crate_pos_features[ACTIONS.index(crate_action)] = 1
            crate_dist_features[ACTIONS.index(crate_action)] = 1
        else:
            crate_dist_features[-1] = 1

    if len(game_state['bombs']) > 0:
        incoming_explosion = [
            deadly_pos
            for bomb_pos, _ in game_state['bombs']
            for deadly_pos in get_bomb_explosion_fields(bomb_pos, game_state['field'])
        ]
        if self_position in incoming_explosion:
            live_saving_action, _ = find_closest_position(self_position, game_state, lambda pos, state: pos not in incoming_explosion)
            if live_saving_action is not None:
                live_saving_features[ACTIONS.index(live_saving_action)] = 1
        else:
            for (nx, ny), action in [((sx - 1, sy), "LEFT"), ((sx + 1, sy), "RIGHT"), ((sx, sy - 1), "UP"), ((sx, sy + 1), "DOWN")]:
                if (nx, ny) in incoming_explosion:
                    deadly_features[ACTIONS.index(action)] = 1
    
    for (nx, ny), action in [((sx - 1, sy), "LEFT"), ((sx + 1, sy), "RIGHT"), ((sx, sy - 1), "UP"), ((sx, sy + 1), "DOWN")]:
        if game_state['explosion_map'][nx, ny] > 0:
            deadly_features[ACTIONS.index(action)] = 1
    
    bomb_explosion_fields = get_bomb_explosion_fields(self_position, game_state['field'])
    _, dist_savety = find_closest_position(self_position, game_state, lambda pos, state: pos not in bomb_explosion_fields)
    bomb_survivable_feature[0] = 1 if dist_savety <= BOMB_TIMER and dist_savety != -1 else 0
    bomb_usefull_feature[0] = 1 if contains_crate(bomb_explosion_fields, game_state['field']) else 0

    return features

# Old state_to_features
# 
# def state_to_features_old(game_state: dict) -> np.array:
#     """
#     *This is not a required function, but an idea to structure your code.*
# 
#     Converts the game state to the input of your model, i.e.
#     a feature vector.
# 
#     You can find out about the state of the game environment via game_state,
#     which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
#     what it contains.
# 
#     :param game_state:  A dictionary describing the current game board.
#     :return: np.array
#     """
#     # This is the dict before the game begins and after it ends
#     if game_state is None:
#         return None
# 
#     # For example, you could construct several channels of equal shape, ...
#     channels = []
#     arena = game_state['field']
#     _, _, _, (pos_x, pos_y) = game_state['self']
#     ox, oy = -pos_x + ROWS - 2, -pos_y + COLS - 2
# 
#     features = np.zeros(FEATURE_SHAPE_OLD)
# 
#     # WALL = -1, FREE = 0, CRATE = 1
#     features[FEAT_WALLS, ox : ox+arena.shape[0], oy : oy+arena.shape[1]] = np.maximum(-arena, 0)
#     features[FEAT_CRATES, ox : ox+arena.shape[0], oy : oy+arena.shape[1]] = np.maximum(arena, 0)
# 
#     for (x, y) in game_state['coins']:
#         features[FEAT_COINS, ox + x, oy + y] = 1
# 
#     for ((x, y), timer) in game_state['bombs']: 
#         features[FEAT_BOMBS, ox + x, oy + y] = 1 + BOMB_TIMER - timer
# 
#     for _, _, _, (x, y) in game_state['others']:
#         features[FEAT_OTHERS, ox + x, oy + y] = 1
# 
#     return features.reshape(-1)
# 