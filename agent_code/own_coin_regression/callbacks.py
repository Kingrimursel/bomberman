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

    self.feature_number = 6
    self.ACTIONS = ACTIONS

    if not os.path.isfile("my-saved-model.pt"):  # TODO: remove the True
        self.logger.info("Setting up model from scratch.")

        # the beta's
        self.model = {  # TODO: initialize at random
                        "UP":    np.random.rand(self.feature_number),
                        "RIGHT": np.random.rand(self.feature_number),
                        "DOWN":  np.random.rand(self.feature_number),
                        "LEFT":  np.random.rand(self.feature_number),
                        "WAIT":  np.random.rand(self.feature_number),
                        "BOMB":  np.random.rand(self.feature_number)
                    }
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


    # Hyperparameters
    self.BATCH_SIZE = 60
    self.alpha      = 0.3
    self.gamma      = 0.1
    self.epsilon    = 0.2


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # exploration
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])  # TODO: allow bombing exploration


    # exploitation
    features = state_to_features(self, game_state)

    self.logger.debug("Querying model for action.")
    return self.ACTIONS[np.argmax(get_Q_values(self, features)[:-1])]


def create_training_batch(self):
    """
        Create Scattermatrix in order to perform PCA there. We hope to get few but meaningfull features this way
        to ensure fast convergence
    """

    features = {
                "UP": [],
                "RIGHT": [],
                "DOWN": [],
                "LEFT": [],
                "WAIT": [],
                "BOMB": []
    }

    rewards = {
                "UP": [],
                "RIGHT": [],
                "DOWN": [],
                "LEFT": [],
                "WAIT": [],
                "BOMB": []
    }

    # build batches
    for transition in self.transitions:
        if transition.state:
            features[transition.action].append(list(state_to_features(self, transition.state)))
            rewards[transition.action].append(transition.reward)

    # center batches
    for action in self.ACTIONS:
        feature_mean = np.mean(features[action], axis=0) if features[action] else np.array([])
        features[action] -= feature_mean

    return features, rewards


def get_Q_values(self, features):
    """
    Function that returns the Q-values to given features
    """

    Q_values = []


    for action in self.ACTIONS:
        Q_values.append(self.model[action] @ features)

    return Q_values



def state_to_features(self, game_state: dict):
    """
    Converts the game state to the input of the model, i.e. a feature vector.
    :param self: Instance on which it is used (agent).
    :param game_state: A dictionary describing the current game board.
    :return: Tuple with 6 entries for the features.
    """
    features = np.zeros(6, dtype=np.int64)

    # This is the dict before the game begins and after it ends.
    if game_state is None:
        return None


    #print(game_state)

    # Gather information about the game state.
    arena           = game_state['field']
    _, _, _, (x, y) = game_state['self']
    others          = [(n, s, b, xy) for (n, s, b, xy) in game_state['others']] # For calculating the number of coins collected yet
    coins           = game_state['coins']
    free_space      = arena == 0 # For the function

    # CURRENT TILE FEATURE (4): Important for later tasks, in coin task always 1.
    features[4] = 1

    # PHASE FEATURE (5): Needed for other features, because of that determined first. More important later.
    totalscore = 0
    for _, s, _, _ in others:
        totalscore += s

    if len(coins) > 0:
        features[5] = 0
    elif totalscore < 9:
        features[5] = 1
    else:
        features[5] = 2

    # NEIGHBOT TILE FEATURE (0-3):
    _, closest_to_coin = look_for_targets(free_space, (x, y), coins, self.logger, dir=True)
    # This is in order (left, right, below, above)
    for num, (i,j) in enumerate([(x+h, y) for h in [-1, 1]] + [(x, y+h) for h in [-1, 1]]):
        # if tile is free
        if arena[i,j] == 0:
            # and if tile is closest to next coin
            if features[5] == 0 and (i,j) == closest_to_coin:
                features[num] = 3
            else:
                features[num] = 0
        # if tile is Wall
        if arena[i,j] == -1:
            features[num] = 1

    return tuple(features)



### boring helpers

def look_for_targets(free_space, start, targets, logger=None, dir=False):
    """
    Find distance to the closest target (target can be specified in use (coin/opponent/crate...))
    Performs a breadth-first search of the reachable free tiles until a special target is encountered.
    :param free_space: Boolean numpy array. True for free tiles and False for obstacles.
    :param start: the coordinate from which to begin the search.
    :param targets: list or array holding the coordinates of all target tiles.
    :param logger: optional logger object for debugging.
    :param dir: Boolean which indicates whether only the distance to best neighbour is required. Else distance to
        and direction of closet target gets calculated.
    :return: Distance to the closest target if dir is True or direction of and distance to closest target
        if dir is False.
    """
    if len(targets) == 0:
        if dir:
            return 17, (0,0)
        return 17

    frontier    = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best        = start
    best_dist   = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    while len(frontier) > 0:

        # Find distance from current position to all targets, track closest
        current = frontier.pop(0)
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d+dist_so_far[current] <= best_dist:
            best      = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break

        # Add unexplored free neighboring tiles to the queue in a random order.
        x, y      = current
        neighbors = [(x, y) for (x, y) in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    # Determine the first step towards the best found target tile.
    current = best
    while True and dir:
        if parent_dict[current] == start:
            return best_dist, current
        current = parent_dict[current]

    return best_dist
