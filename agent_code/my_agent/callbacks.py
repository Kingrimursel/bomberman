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
        #self.model = np.ones((10, 10, 10, 10, 10, 6))  # one dimension for each feature and one for the actions

        self.category_sizes = np.array([3, 6, 3])  # classifying, defensive, offensive
        self.num_features  = np.sum(self.category_sizes)*5

        self.model = np.ones((self.num_features, 6))

        """self.model = np.ones((
            # classifying
            3, 3, 3, 3, 3,
            # defensive
            6, 6, 6, 6, 6,
            # offensive
            3, 3, 3, 3, 3,
            # explorative

            # actions
            6), dtype=np.float32)"""

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    # TODO: tune these hyper parameters!
    self.alpha = .4
    self.gamma = .2
    self.exploration_prob = .1


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

    get_Q_values(self, state_to_features(self, game_state))

    return ACTIONS[np.argmax(self.model[state_to_features(self, game_state)])]


def get_Q_values(self, features):
    """
    Extract Q-Value from Q-Table corresponding to feature value

    INPUT:
        self: agent
        features: the features
    OUTPUT:
        Q-values corresponding to features

    """

    Q_values = np.empty((len(features), 6))

    #print(Q_values.shape)
    #print(features)

    # shape of features: 5*classifying, 5*defensive, 5*offensive

    # shape of model: 5*3 (classifying), 5*6 (defensive), 5*3 (offfensive)


    # loop over category
    for i, feature in enumerate(features):
        # get category
        category = i // 5

        # get size of category
        print(i, category)
        category_size = self.category_sizes[category]

        # get field (left, right, top etc.)
        field = i % 5

        # get features value position. First skipt the previous categories, then the previous fields.
        # Then add the feature value
        value_position = 5*np.sum(self.category_sizes[:category]) + field*category_size + feature

        #print(category, category_size, field)
        #print(value_position)
        #print(self.model.shape)
        #print(self.model[value_position])

        Q_values[i] = self.model[value_position]


        # i % 5  = category                            (check)
        # self.category_sizes[i%5] = size of category  (check)
        # i // 5 = field                               (check) 

        # value position = 5*np.sum(self.category_sizes[:category]) + size of category * field + feature     
        # feature value  = self.model[value position]

    print(Q_values)




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
    # TODO: currently the features shadow each other. I should probably create new feature for each tile and each current feature?


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


    features = np.empty(5*len(self.category_sizes), dtype=int)

    for i, (x, y) in enumerate([(xs, ys), (xs-1, ys), (xs+1, ys), (xs, ys-1), (xs, ys+1)]):
        # classifying
        features[i] = game_state["field"].T[x, y] + 1

        # defensive
        if future_explosion_map[x, y] != 0:
            features[i + 5] = future_explosion_map[x, y]
        elif game_state["explosion_map"].T[x, y] != 0:
            features[i + 5] = 4
        else:
            features[i + 5] = 5

        # offensive
        if opponents.size != 0 and (x, y) in list(opponents[:, -1]):
            features[i + 10] = 0
        elif (x, y) in list(game_state["coins"]):
            features[i + 10] = 1
        else:
            features[i + 10] = 2

    # save future_explosion_map to game_state
    game_state["future_explosion_map"] = future_explosion_map  # TODO not working!

    return tuple(features)


def get_field_value(i, j, game_state, future_explosion_map, opponents):
    """

    classify field (i, j) with its corresponding feature value

    INPUT:
        i,j:        coordinates on map
        game_state: current game state
        bombs:      numpyified bombs array
        opponents:  numpyified opponents array
    OUTPUT:
        field classification with corresponding value

    """

    ## classifying
    # wall: 0
    # free: 1
    # crate: 2


    ## defensive
    # future explosion: countdown in [0, 3]
    # present explosion: 4
    # no explosion:5

    ## offensive
    # opponent: 0
    # coin: 1
    # nothing: 2

    # TODO: add
    # TODO: unterscheiden zwischen coin und enemy!! maybe a mode which the agent can enter?

    ## explorative 
    # towards next coin: 1
    # towards next enemy: 2
    # away from both: 3


    # caution: order of lookup matters!

    # future explosion
    if future_explosion_map[i, j] > 0:
        return future_explosion_map[i, j] + 2

    # current explosion
    elif game_state["explosion_map"].T[i, j] == 1:
        return 6

    # opponent
    elif opponents.size != 0 and (i, j) in list(opponents[:, -1]):
        return 7

    # coin
    elif (i, j) in list(game_state["coins"]):
        return 8

    # wall, crate, empty
    return game_state["field"].T[i, j]


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

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
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


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

    arena_abs = np.abs(arena)

    future_explosion_map = np.zeros((width, height), dtype=int)

    # remember where non-free tiles are, since explosions get stopped there. removed later
    future_explosion_map[arena != 0] = -10  # TODO still necessary with wall_inbetween?

    # loop over all currently placed bombs
    for (y, x), t in bombs:  # transposing once again
        for i in range(4):  # TODO: why do I have to transpose here? Check this!
            if x - i > 0 and not wall_inbetween(x-i, y, x, y, arena_abs):
                future_explosion_map[x-i, y] = t
            if x + i < width - 1 and not wall_inbetween(x+i, y, x, y, arena_abs):
                future_explosion_map[x+i, y] = t
            if y - i > 0 and not wall_inbetween(x, y-i, x, y, arena_abs):
                future_explosion_map[x, y-i] = t
            if y + i < height - 1 and not wall_inbetween(x, y+i, x, y, arena_abs):
                future_explosion_map[x, y+i] = t


    # remove safe tiles again
    future_explosion_map[future_explosion_map < 0] = 0

    return future_explosion_map


def wall_inbetween(x1, y1, x2, y2, arena_abs):
    """
    Checks if there is a wall inbetween (x1, y1) and (x2, y2)


    INPUT:
        xi, yi: coordinates
        arena_abs: np.abs of field

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

    return np.sum(arena_abs[slice(x_min, x_max), slice(y_min, y_max)]) != 0




def get_danger_level(i, j, future_explosion_map):
    """
    returns danger level on tile (i,j). Only determined by the bomb timer on this tile

    INPUT:
        i,j : position on map


    OUTPUT:
        possible bomb timer at position i,j

    """

    # TODO: add current_explosions_map lookup

    return future_explosion_map[i, j]


