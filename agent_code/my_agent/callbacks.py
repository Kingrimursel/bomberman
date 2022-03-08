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

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        self.model = np.ones((
            # defensive
            7, 7, 7, 7, 7,
            # offensive
            3, 3, 3, 3, 3,
            # explorative

            # actions
            6), dtype=np.float32)

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    # TODO: tune these hyper parameters!
    self.alpha = .4
    self.gamma = .2
    self.exploration_prob = .1


def look_for_targets(free_space, start, targets, logger=None):
    """Find distance to the closest target (target can be specified in use)
    Performs a breadth-first search of the reachable free tiles until a special target is encountered.
    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        distance to the closest target
    """

    if len(targets) == 0:
        return 17

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    #if logger: logger.debug(f'Suitable target found at {best}')
    #if logger: logger.debug(f'Suitable target found with distance {best_dist}')

    # TODO: strecke mitnehmen, sodass man nicht für jeden nachbarn die distanz erneut berechnen muss.

    return best_dist


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # extract features from state
    features = state_to_features(self, game_state)

    if self.train and random.random() < self.exploration_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    return ACTIONS[np.argmax(self.model[state_to_features(self, game_state)])]


def state_to_features(self, game_state: dict) -> np.array:
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
    # the cites can take different values, depending on the circumstances like type etc.

    # TODO: add map awareness

    # TODO: add feature that shows you to next coin. I.e. if moving to a tile brings you closer to
    # the coin, give it a higher value
    # TODO: add feature that shows you to next opponent
    # TODO: add looking for next crate

    # TODO: reward step into direction of coin

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    ## utilities and declarations

    # numpyify the two lists for better access
    bombs = np.array(game_state["bombs"], dtype=object)  # TODO explicit definition not necessary anymore
    opponents = np.array(game_state["others"], dtype=object)

    # coordinates of agents current position
    xs,ys = game_state["self"][-1]


    # compute future explosion map
    future_explosion_map = create_future_explosion_map(bombs, game_state["field"].T)

    # compute distance to next coin
    free_space = game_state["field"].T == 0

    d_next_coin     = look_for_targets(free_space, (xs, ys), game_state["coins"], self.logger)
    #d_next_opponent = look_for_targets(free_space, (xs, ys), opponents[:, -1]   , self.logger)


    # actually calculate features
    num_features = len(np.array(self.model.shape[:-1]))
    features = np.empty(num_features, dtype=int)

    width, height = game_state["field"].shape
    for i, (x, y) in enumerate([(xs, ys), (xs-1, ys), (xs+1, ys), (xs, ys-1), (xs, ys+1)]):

        # if we are in the boundary
        if x >= 0 and x < width - 1 and y >= 0 and y < height - 1:
            updated_d_next_coin     = look_for_targets(free_space, (x, y), game_state["coins"], self.logger)
            #updated_d_next_opponent = look_for_targets(free_space, (x, y), opponents[:, -1]   , self.logger)
        else:
            updated_d_next_coin     = 1000
            #updated_d_next_opponent = 1000

        # defensive
        if future_explosion_map[x, y] != 0:
            features[i] = future_explosion_map[x, y] - 1  # start at zero to save memory
        elif game_state["explosion_map"].T[x, y] != 0:
            features[i] = 3
        else:
            features[i] = game_state["field"].T[x, y] + 5

        # offensive
        if opponents.size != 0 and (x, y) in list(opponents[:, -1]):
            features[i + 5] = 0
        elif updated_d_next_coin < d_next_coin:
            features[i + 5] = 1
        else:
            features[i + 5] = 2


    # save future_explosion_map to game_state
    game_state["future_explosion_map"] = future_explosion_map  # TODO not working!

    return tuple(features)


def create_future_explosion_map(bombs, arena):
    """
        Create map of future explisions

        INPUT:
            bombs: np.array of currently placed bombs
            arena: current map
        OUTPUT:
            map of bombs and timers
    """

    width, height = arena.shape

    arena_walls = arena.copy()
    arena_walls[arena_walls != -1] = 0

    future_explosion_map = np.zeros((width, height), dtype=int)

    # remember where non-free tiles are, since explosions get stopped there. removed later
    future_explosion_map[arena != 0] = -10  # TODO still necessary with wall_inbetween?

    # loop over all currently placed bombs
    for (y, x), t in bombs:  # transposing once again
        for i in range(4):
            if x - i > 0 and not wall_inbetween(x-i, y, x, y, arena_walls):
                future_explosion_map[x-i, y] = t
            if x + i < width - 1 and not wall_inbetween(x+i, y, x, y, arena_walls):
                future_explosion_map[x+i, y] = t
            if y - i > 0 and not wall_inbetween(x, y-i, x, y, arena_walls):
                future_explosion_map[x, y-i] = t
            if y + i < height - 1 and not wall_inbetween(x, y+i, x, y, arena_walls):
                future_explosion_map[x, y+i] = t


    # remove safe tiles again
    future_explosion_map[future_explosion_map < 0] = 0

    return future_explosion_map


def wall_inbetween(x1, y1, x2, y2, arena_walls):
    """
    Checks if there is a wall inbetween (x1, y1) and (x2, y2)


    INPUT:
        xi, yi: coordinates
        arena_walls: np.abs of field

    OUTPUT:
        boolean indicating whether there is a wall inbetwee the two tiles 
    """

    # bombs only explode along one coordinate axis. simulate this by returning False (for safety)
    if x1!=x2 and y1!=y2:
        return False

    # array slicing indices
    x_min = min(x1, x2)
    x_max = max(x1, x2) + 1
    y_min = min(y1, y2)
    y_max = max(y1, y2) + 1

    return np.sum(arena_walls[slice(x_min, x_max), slice(y_min, y_max)]) != 0




def get_danger_level(i, j, future_explosion_map, game_state):
    """
    returns danger level on tile (i,j). Only determined by the bomb timer on this tile

    INPUT:
        i,j : position on map


    OUTPUT:
        possible bomb timer at position i,j

    """

    # TODO: add current_explosions_map lookup

    if game_state["explosion_map"].T[j, i] != 0:
        return -1

    return future_explosion_map[j, i]
