import os
import pickle
import random
import numpy as np

from random import shuffle
from collections import namedtuple, deque
from sklearn.ensemble import GradientBoostingRegressor

from agent_code.rule_based_agent import act as act_determinstic

# Additional structures
ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'])
ACTION_TO_INDEX = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
}

# Hyperparameter
NAME           = "my-saved-model"
INITIAL_ROUNDS = 1
PARAMETER_FOR_GBR = {
    'warm_start': True,
    'n_estimators': 10,
    'learning_rate': 0.6,
    'max_depth': 20,
}

def setup(self):
    """
    Setup your code. This is called once when loading each agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.code_name   = NAME

    # NOTE: Each time agent is trained, model from scratch is set up.
    if self.train or not os.path.isfile(self.code_name + ".pt"):
        self.logger.info("Setting up model from scratch.")
        self.initial_rounds = INITIAL_ROUNDS
        self.model = GradientBoostingRegressor(**PARAMETER_FOR_GBR)
    else:
        self.logger.info("Loading model from saved state.")
        self.initial_rounds = 0
        with open(self.code_name + ".pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Decides which action to take.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.initial_rounds > 0:
        return act_determinstic(self, game_state)

    if self.train and random.random() < self.epsilon:
        return np.random.choice(ACTIONS)

    features_exept_actions = np.repeat(np.array(state_to_features(self, game_state))[None], 5, axis=0)
    actions                = np.arange(5)
    actions.resize((5, 1))
    features               = np.append(features_exept_actions, actions, axis=1)
    predictions            = self.model.predict(features)

    # Avoiding loops if np.argmax is not unique
    if np.sum(predictions == predictions[np.argmax(predictions)]) != 1:
        return np.random.choice(ACTIONS[predictions == predictions[np.argmax(predictions)]])
    else:
        return ACTIONS[np.argmax(predictions)]


def state_to_features(self, game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e. a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None

    # Initialize array and define normalization.
    feature = np.empty(5)
    norm    = np.array([2, 2, 2, 2, 4])/100

    # FEATURE 0-3: Are tiles around agent free? Order: Up, Right, Down, Left.
    position_x, position_y = game_state['self'][3]
    feature[0] = game_state['field'][(position_x, position_y-1)] + 1
    feature[1] = game_state['field'][(position_x+1, position_y)] + 1
    feature[2] = game_state['field'][(position_x, position_y+1)] + 1
    feature[3] = game_state['field'][(position_x-1, position_y)] + 1

    # FEATURE 4: In which direction is the next nearest coin?
    self_position = np.copy(np.array(game_state['self'][3]))
    step_to_coin  = np.array(step_to_targets(game_state['field'] == 0, game_state['self'][3], game_state['coins']))

    try:
        direction = step_to_coin - self_position
    except:
        return np.zeros(5)

    assert np.sum(np.abs(direction)) == 1, f"Next step to nearest coin is not next to self_position: {direction}"
    if direction[1] == -1:
        feature[4] = 0
    elif direction[0] == 1:
         feature[4] = 1
    elif direction[1] == 1:
        feature[4] = 2
    else:
        feature[4] = 3

    return feature/norm


def step_to_targets(free_space, start, targets, logger=None) -> tuple:
    """
    Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    :param free_space: Boolean numpy array. True for free tiles and False for obstacles.
    :param start: the coordinate from which to begin the search.
    :param targets: list or array holding the coordinates of all target tiles.
    :param logger: optional logger object for debugging.
    :return: coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    if start in targets:
        targets.remove(start)

    frontier    = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best        = start
    best_dist   = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)

        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best      = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break

        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    if logger:
        logger.debug(f'Suitable target found at {best}')

    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def look_for_targets(free_space: np.array, start: tuple, targets: list, logger=None) -> tuple:
    """
    Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    :param free_space: Boolean numpy array. True for free tiles and False for obstacles.
    :param start: the coordinate from which to begin the search.
    :param targets: list or array holding the coordinates of all target tiles.
    :param logger: optional logger object for debugging.
    :return: coordinate of closest target.
    """
    if len(targets) == 0:
        return None

    if start in targets:
        targets.remove(start)

    frontier    = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best        = start
    best_dist   = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)

        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best      = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break

        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    if logger:
        logger.debug(f'Suitable target found at {best}')

    return best
