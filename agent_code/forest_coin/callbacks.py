import os
import pickle
import random
import numpy as np

from random import shuffle
from collections import namedtuple, deque
from sklearn.ensemble import GradientBoostingRegressor

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
    'max_depth': 30,
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
    actions = np.arange(5)
    actions.resize((5, 1))
    features = np.append(features_exept_actions, actions, axis=1)
    predictions = self.model.predict(features)

    # Avoiding loops if np.argmax is not unique
    if np.sum(predictions == predictions[np.argmax(predictions)]) != 1:
        action = np.random.choice(ACTIONS[predictions == predictions[np.argmax(predictions)]])
        return action
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


def act_determinstic(self, game_state: dict) -> str:
    """
    Called each game step to determine the agent's next action.

    :param self: Self object of Class Agend.
    :param game_stat: Current game state.
    """
    self.logger.info('Picking action according to rule set')
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if bombs_left > 0:
        valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates

    # Exclude targets that are currently occupied by a bomb
    targets = [target for target in targets if target not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    for o in others:
        free_space[o] = False
    d = step_to_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) <= s.BOMB_POWER):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) <= s.BOMB_POWER):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            return a
