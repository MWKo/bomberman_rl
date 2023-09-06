import os
import random
import pickle

import numpy as np
from .constants import *
from settings import BOMB_TIMER, ROWS, COLS

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile(MODEL_FILE_NAME):
    #if not os.path.isfile(MODEL_FILE_NAME):
        self.logger.info(MODEL_FILE_NAME)
        weights = np.random.rand(FEATURE_SIZE, len(ACTIONS))
        self.model = weights / np.abs(weights).sum(axis=0)
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE_NAME, "rb") as file:
            self.model = pickle.load(file)
        # print(self.model)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.20, .20, .20, .20, .1, .1])

    self.logger.debug("Querying model for action.")
    q_vector = self.model.T @ state_to_features(game_state)
    action = ACTIONS[np.argmax(q_vector)]

    """
    if not self.train:
        print(action)
        print(state_to_features(game_state))
        print(q_vector)
        print()
    """

    return action


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
        if arena[x + dx, y + dy] == -1:
            continue
        explosion_fields.append((x + dx, y + dy))
        explosion_fields.append((x + dx * 2, y + dy * 2))
        explosion_fields.append((x + dx * 3, y + dy * 3))
    return explosion_fields

def find_closest_position_action(game_state: dict, is_searched_position):
    _, _, _, self_position = game_state['self']
    arena = game_state['field']
    queue = [(self_position, 'WAIT', 0)]
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
            if arena[nx, ny] != 0 or visited[nx, ny]:
                continue
            queue.append(((nx, ny), naction if action == "WAIT" else action, dist + 1))
    
    return None, -1


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None
    
    _, _, _, self_position = game_state['self']
    sx, sy = self_position  
    
    features = np.zeros(FEATURE_SIZE)
    coin_features = features[COIN_FEATURES_START : COIN_FEATURES_START + COIN_FEATURES_LENGTH]
    crate_features = features[CRATE_FEATURES_START : CRATE_FEATURES_START + CRATE_FEATURES_LENGTH]
    live_saving_features = features[LIVE_SAVING_FEATURES_START : LIVE_SAVING_FEATURES_START + LIVE_SAVING_FEATURES_LENGTH]
    deadly_features = features[DEADLY_FEATURES_START : DEADLY_FEATURES_START + DEADLY_FEATURES_LENGTH]
    bomb_survivable_feature = features[BOMB_SURVIVABLE_FEATURES_START : BOMB_SURVIVABLE_FEATURES_START + BOMB_SURVIVABLE_FEATURES_LENGTH]
    
    coin_action, _ = find_closest_position_action(game_state, lambda pos, state: pos in state['coins'])
    if coin_action is not None:
        coin_features[ACTIONS.index(coin_action)] = 1

    crate_action, dist = find_closest_position_action(game_state, 
        lambda pos, state: is_next_to(pos, lambda p: state['field'][p[0], p[1]] == 1)
    )
    if crate_action is not None:
        crate_features[ACTIONS.index(crate_action)] = 1
        crate_features[-1] = 1 if dist == 0 else 0

    if len(game_state['bombs']) > 0:
        incoming_explosion = [
            deadly_pos
            for bomb_pos, _ in game_state['bombs']
            for deadly_pos in get_bomb_explosion_fields(bomb_pos, game_state['field'])
        ]
        if self_position in incoming_explosion:
            live_saving_action, _ = find_closest_position_action(game_state, lambda pos, state: pos not in incoming_explosion)
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
    _, dist_savety = find_closest_position_action(game_state, lambda pos, state: pos not in bomb_explosion_fields)
    bomb_survivable_feature[0] = 1 if dist_savety <= BOMB_TIMER and dist_savety != -1 else 0


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