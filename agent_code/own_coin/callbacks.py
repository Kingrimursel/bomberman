import os
import sys
import pickle
import random

from collections import deque
from random import shuffle
import numpy as np

sys.path.append(os.path.abspath(".."))

from agent_code.own_coin import config


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Constants
NAME = "my-saved-model"

class color:
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    NC='\033[0m'


def setup(self):
    """
    Setup your code. This is called once when loading each agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.code_name = NAME

    if not os.path.isfile(self.code_name + ".pt"): #if no model saved
        self.logger.info("Setting up model from scratch.")
        self.model = np.ones((4,4,4,4,5,3,6))*[.25, .25, .25, .25, .0, .0] #Initial guess

    else: #if model saved, no matter if in train mode or not, load current model #TODO: Why is this not done in the given code? In training mode a random model is picked always
        self.logger.info("Loading model from saved state.")
        with open(self.code_name + ".pt", "rb") as file:
            self.model = pickle.load(file)

    print(f"TRULY_TRAIN={color.RED}{config.TRULY_TRAIN}{color.NC}")
    print(f"DETERMINISTIC={color.RED}{config.DETERMINISTIC}{color.NC}")
    print(f"EPSILON={color.RED}{config.EPSILON}{color.NC}")

    if config.DETERMINISTIC:
        np.random.seed()
        # Fixed length FIFO queues to avoid repeating the same actions
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0
        self.current_round = 0

    # hyperparameters
    self.alpha = config.ALPHA
    self.gamma = config.GAMMA
    self.epsilon = config.EPSILON

    # for keeping track of a game store
    self.score = {}


def act(self, game_state: dict):
    """
    Agent parses the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if config.DETERMINISTIC:
        return deterministic_action(self, game_state)

    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0, .0])

    self.logger.debug("Querying model for action.")

    features = state_to_features(self, game_state) #get features from game_state

    return ACTIONS[np.argmax(self.model[features][:-1])] #Gives action with maximal reward for given state, never returns 'BOMB'


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

### boring functions

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def look_for_targets_original(free_space, start, targets, logger=None):
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


def deterministic_action(self, game_state):
    """
        Determine next step deterministcally. Used for increased learning speed.
    """

    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
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

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

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
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
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
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets_original(free_space, (x, y), targets, self.logger)
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
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
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
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a
