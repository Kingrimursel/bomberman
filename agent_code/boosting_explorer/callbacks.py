import os
import pickle
import random
import numpy as np

from random import shuffle
from collections import namedtuple, deque
from sklearn.ensemble import GradientBoostingRegressor

from .callbacks_own_explorer import setup as setup_own_explorer
from .callbacks_own_explorer import act as act_own_explorer

from agent_code.rule_based_agent import callbacks as rule_based

# Additional structures
ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])
ACTION_TO_INDEX = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
    'BOMB': 5,
}

# Hyperparameter
NAME              = "test-model"
PARAMETER_FOR_GBR = {
    'warm_start': True,
    'n_estimators': 0,
    'learning_rate': 0.8,
    'max_depth': 20,
}

def setup(self):
    """
    Setup your code. This is called once when loading each agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.code_name = NAME
    rule_based.setup(self)

    # NOTE: Each time agent is trained, model from scratch is set up.
    if self.train or not os.path.isfile(self.code_name + ".pt"):
    #if not os.path.isfile(self.code_name + ".pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = GradientBoostingRegressor(**PARAMETER_FOR_GBR)
    else:
        self.logger.info("Loading model from saved state.")
        self.rounds_initial = 0
        with open(self.code_name + ".pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Decides which action to take.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.train:
        self.logger.debug(f"Before new action: \nself.rounds_initial: {self.rounds_initial}\nbatch_count: {self.batch_count}\nn_rounds: {game_state['round']}")

    if self.train and self.rounds_count <= self.rounds_initial:
        return rule_based.act(self, game_state)

    if self.train and random.random() < self.epsilon:
        return np.random.choice(np.array(ACTIONS), p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.75])

    features_except_actions = np.repeat(np.array(state_to_features(self, game_state))[None], 6, axis=0)
    actions                 = np.arange(6)
    actions.resize((6, 1))
    features                = np.append(features_except_actions, actions, axis=1)
    predictions             = self.model.predict(features)

    self.logger.debug(f"Step {game_state['step']}: feature_except_action: {features_except_actions[0]}\npredictions {predictions}")

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
    feature         = np.zeros(8, dtype=np.double)
    arena           = game_state['field']
    danger_arena    = danger(game_state)
    no_danger_arena = np.logical_not(danger_arena)
    norm            = np.array([4, 4, 4, 4, 4, 5, 5, 5])/100

    self_position  = game_state['self'][3]
    self_x, self_y = self_position

    # FEATURE 0-4: Are tiles around agent free or is there danger? Order: Up, Right, Down, Left, Origin.
    for i, tile in enumerate( [(self_x + h_x, self_y + h_y) for h_x, h_y in [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]] ):
        feature[i] = game_state['field'][tile] + 1
        if danger_arena[tile] and game_state['field'][tile] == 0:
            feature[i] = 3

    # FEATURE 5: In which direction is the next nearest coin?
    step_to_coin  = np.array(step_to_targets(np.logical_and(arena == 0, no_danger_arena), self_position, game_state['coins']))
    direction     = step_to_coin - self_position

    if direction[1] == -1:
        feature[5] = 0
    elif direction[0] == 1:
        feature[5] = 1
    elif direction[1] == 1:
        feature[5] = 2
    elif direction[0] == -1:
        feature[5] = 3
    elif np.sum(direction) == 0:
        feature[5] = 4
    else:
        raise ValueError("Direction is incorrect.")

    # FEATURE 6: In which direction to escape the danger?
    step_to_escape = np.array(step_to_targets(arena == 0, self_position, np.argwhere(no_danger_arena)))
    direction      = step_to_escape - self_position

    if direction[1] == -1:
        assert direction[0] == 0, "Position of Agent is expected to be on the danger cross."
        feature[6] = 0
    elif direction[0] == 1:
        assert direction[1] == 0, "Position of Agent is expected to be on the danger cross."
        feature[6] = 1
    elif direction[1] == 1:
        assert direction[0] == 0, "Position of Agent is expected to be on the danger cross."
        feature[6] = 2
    elif direction[0] == -1:
        assert direction[1] == 0, "Position of Agent is expected to be on the danger cross."
        feature[6] = 3
    elif np.sum(direction) == 0:
        feature[6] = 4
    else:
        raise ValueError("Direction is incorrect.")

    # FEATURE 7: Where to go for next crate?
    step_to_crate = np.array(step_to_targets(np.logical_and(arena == 0, no_danger_arena), self_position, np.argwhere(np.logical_and(arena == 1, no_danger_arena))))
    direction     = step_to_crate - self_position

    if direction[1] == -1:
        assert direction[0] == 0, "Direction is expected to be an array pointing upward, at the left, downwards or at the right."
        feature[7] = 0
    elif direction[0] == 1:
        assert direction[1] == 0, "Direction is expected to be an array pointing upward, at the left, downwards or at the right."
        feature[7] = 1
    elif direction[1] == 1:
        assert direction[0] == 0, "Direction is expected to be an array pointing upward, at the left, downwards or at the right."
        feature[7] = 2
    elif direction[0] == -1:
        assert direction[1] == 0, "Direction is expected to be an array pointing upward, at the left, downwards or at the right."
        feature[7] = 3
    elif np.sum(direction) == 0:
        feature[7] = 4
    else:
        raise ValueError("Direction is incorrect.")

    return feature/norm


def step_to_targets(free_space: np.array, start: tuple, targets, logger=None) -> tuple:
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
        return start

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


def look_for_targets(free_space: np.array, start: tuple, targets, logger=None) -> tuple:
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
        return start

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


def danger(game_state: dict) -> np.array:
    """
    Determines for each tile the potential of danger.

    :param game_state: The current game state.
    :return: Boolean map of danger.
    """
    bombs  = game_state['bombs']
    arena  = game_state['field']
    danger = game_state['explosion_map'].astype(bool)

    for (x, y), _ in bombs:
        danger[x, y] = True
        for direction_x, direction_y in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            d = 1
            while arena[x + direction_x*d, y + direction_y*d] != -1 and d < 4:
                danger[x + direction_x*d, y + direction_y*d] = True
                d += 1

    return danger
