import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    # extract features from state
    features = state_to_features(game_state)

    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


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

    # we consider five features: one for each neighbouring cite and one for the current one.
    # the cites can take different values, depending on the circumstances.

    # wall: -1
    # free: 0
    # crate: 1
    # bomb: countdown + 2 in [2, 5]
    # smoke: 6
    # opponent: 7
    # coin: 8

    bombs    = np.array(game_state["bombs"], dtype=object)
    oponents = np.array(game_state["others"], dtype=object)

    x,y = game_state["self"][-1]

    current_field  = get_field_value(x,   y,   game_state, bombs, oponents)
    left_field     = get_field_value(x-1, y,   game_state, bombs, oponents)
    right_field    = get_field_value(x+1, y,   game_state, bombs, oponents)
    top_field      = get_field_value(x,   y-1, game_state, bombs, oponents)
    bottom_field   = get_field_value(x,   y+1, game_state, bombs, oponents)


    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


def get_field_value(i, j, game_state, bombs, oponents):
    """

    classify field (i, j)

    """

    # caution: order of lookup matters!

    # bomb
    if len(bombs) > 0 and (i, j) in list(bombs[:, 0]):
        return bombs[list(bombs[:, 0]).index((i, j))][1] + 2

    # smoke
    elif game_state["explosion_map"][i, j] == 1:
        return 6

    # opponent
    elif (i, j) in list(oponents[:, -1]):
        return 7

    # coin
    elif (i, j) in list(game_state["coins"]):
        return 8


    # wall, crate, empty
    return game_state["field"][i, j]

