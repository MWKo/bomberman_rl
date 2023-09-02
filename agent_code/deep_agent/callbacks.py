import os
import random

import numpy as np
import torch
from .constants import *
from .neural_net import NeuralNetwork, Trainer, device
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
    
    if self.train:
        self.trainer = Trainer(
            model_file_name=MODEL_FILE_NAME,
            loss_fn=torch.nn.HuberLoss(),
            optimizer_constructor=lambda params: torch.optim.Adam(params, lr=0.03),
            gamma=0.99
        )
        self.forward = self.trainer.forward
    else:
        self.model = NeuralNetwork().to(device)
        if os.path.isfile(MODEL_FILE_NAME):
            self.model.load_state_dict(torch.load("deep-model.pt"))
        self.forward = self.model.forward
    # self.model.eval()

    """
    if self.train:
        self.model.train()
    else:
        self.model.eval()
        """

    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    """


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .3
    q_vector = self.forward(state_to_features(game_state))

    #print(q_vector)

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0, .0])

    self.logger.debug("Querying model for action.")
    action = ACTIONS[torch.argmax(q_vector)]
    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    arena = game_state['field']
    _, _, _, (pos_x, pos_y) = game_state['self']
    ox, oy = -pos_x + ROWS - 2, -pos_y + COLS - 2

    features = np.zeros(FEATURE_SHAPE)

    # WALL = -1, FREE = 0, CRATE = 1
    features[FEAT_WALLS, ox : ox+arena.shape[0], oy : oy+arena.shape[1]] = np.maximum(-arena, 0)
    features[FEAT_CRATES, ox : ox+arena.shape[0], oy : oy+arena.shape[1]] = np.maximum(arena, 0)

    for (x, y) in game_state['coins']:
        features[FEAT_COINS, ox + x, oy + y] = 1

    for ((x, y), timer) in game_state['bombs']: 
        features[FEAT_BOMBS, ox + x, oy + y] = 1 + BOMB_TIMER - timer

    for _, _, _, (x, y) in game_state['others']:
        features[FEAT_OTHERS, ox + x, oy + y] = 1

    return torch.tensor(features.reshape(-1), dtype=torch.float).double().to(device=device)
