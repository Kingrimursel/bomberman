import os
import pickle
import random

import numpy as np

# Addition structures.
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Constants
NAME = "my-saved-model"

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


def act(self, game_state: dict):
    """
    Agent parses the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #exponential exploration/exploitation
    # if(game_state['round']>1):
    #     random_prob = np.exp(-0.01*game_state['round'])
    # else:

    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80% walk in any direction. wait 20%. Bomb 0%
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0, .0])

    self.logger.debug("Querying model for action.")

    features = state_to_features(self, game_state) #get features from game_state

    self.logger.info(f"FEATURE CALCULATED")
    self.logger.info(f"Features: {features}")
    #return np.random.choice(ACTIONS, p=self.model[0])

    return ACTIONS[np.argmax(self.model[features])] #Gives action with maximal reward for given state


# The following two functions are defined in here so they can also be used in the train.py.
def look_for_targets(free_space, start, targets, logger=None, dir=False):
    """
    Find distance to the closest target (target can be specified in use (coin/opponent/crate...))

    Performs a breadth-first search of the reachable free tiles until a special target is encountered.
    :param free_space: Boolean numpy array. True for free tiles and False for obstacles.
    :param start: the coordinate from which to begin the search.
    :param targets: list or array holding the coordinates of all target tiles.
    :param logger: optional logger object for debugging.
    :return: distance to the closest target or direction of closest target
    """
    if len(targets) == 0:
        if dir: return 17, (0,0)
        return 17

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
    #if logger: logger.debug(f'Suitable target found with distance {best_dist}')
    # Determine the first step towards the best found target tile
    current = best

    while True and dir:
        if parent_dict[current] == start: return best_dist,current
        current = parent_dict[current]

    return best_dist


def state_to_features(self, game_state: dict) -> np.array:
    """
    Converts the game state to the input of the model, i.e. a feature vector.

    :param self: Instance on which it is used (agent).
    :param game_state: A dictionary describing the current game board.
    :return: Tuple with 6 entries for the features.
    """
    features = np.zeros(6, dtype=np.int64)

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Gather information about the game state.
    arena        = game_state['field']
    step         = game_state['step']
    n,s,b,(x, y) = game_state['self']
    others       = [(n, s, b, xy) for (n, s, b, xy) in game_state['others']] #For calculating the number of coins collected yet
    coins        = game_state['coins']

    cols       = range(1, arena.shape[0] - 1)
    rows       = range(1, arena.shape[0] - 1)
    walls      = [(x, y) for x in cols for y in rows if (arena[x, y] == -1)]
    free_tiles = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)]
    free_space = arena == 0 #For the function

    #Phase Feature (5) (Needed for other features, because of that determined first)
    #Determine total number of found coins
    total_agents = 0
    if step == 1:
        total_agents = len(game_state['others'])

    totalscore = 0
    for n,s,b,(xo,yo) in others:
        totalscore += s
    totalcoins = totalscore - (total_agents - len(others))*5

    if len(coins) > 0:
        features[5] = 0
    elif totalcoins < 9:
        features[5] = 1
    else:
        features[5] = 2

    #Neighbor tile features (0-3)
    #All the three are needed for the phase dependent decision
    distance_nextcoin, closest_to_coin = look_for_targets(free_space, (x, y), coins, self.logger, dir=True) #distance to closest coin
    #This is in order (Left, Right, Above, Below)
    for num, (i,j) in enumerate([(x+h, y) for h in [-1, 1]] + [(x, y+h) for h in [-1, 1]]):
        #if tile is free
        if arena[i,j] == 0:
            features[num] = 0
        # if tile is Wall
        if arena[i,j] == -1:
            features[num]=1
        # possible danger (a bomb in the area may explode soon)
        elif arena[i,j] == 0 and False: #Value for later with opponents
            features[num]=2
        #Phase dependent value
        else:
            if features[5] == 0:
                #When tile is closest to next coin, mark with this value
                if (i,j) == closest_to_coin:
                    features[num]=3
            # The following cases do not occur in this game mode, they are for later
            elif features[5] == 1:
                features[num] = 3
            elif features[5] == 2 :
                features[num] = 3

    #Feature for current tile (4) (Also more important for later tasks, in coin task always 1)
    features[4] = 1
    #wait does not lead to sure death and (can not place bomb (current field is bomb or bomb was placed recently) or placing bomb leads to sure death(trapped))
    # if(bomb_map[x,y]>0 and (bomb_positions[x,y]==1 or features[0]==features[1]==features[2]==features[3]==1 or not b)):
    #     features[4]=0
    # #not (directly) trapped and the possible destruction is (0-2)/(3-5)/(>6)
    # elif(features[0]!=1 and features[1]!=1 and features[2]!= 1 and features[3]!=1 and ( possible_destruction < 3)):
    #     features[4]=1
    # elif(features[0]!=1 and features[1]!=1 and features[2]!= 1 and features[3]!=1 and (3 <= possible_destruction < 6)):
    #     features[4]=2
    # elif(features[0]!=1 and features[1]!=1 and features[2]!= 1 and features[3]!=1 and (6 <= possible_destruction)):
    #     features[4]=3
    # elif(bomb_map[x,y]==0 or bomb_positions[x,y]==1): #Bomb placed on current tile or wait is safe death
    #     features[4]=4

    output = tuple(features)
    #self.logger.debug(f'Features Calculated: {output}')
    return output
