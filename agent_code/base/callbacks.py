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
    that are independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not os.path.isfile("my-saved-model.pt"): #if no model saved
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = np.ones((4,4,4,4,5,3,6))*[.25, .25, .25, .25, .0, .0] #Initial guess

    else: #if model saved, no matter if in train mode or not, load current model #TODO: Why is this not done in the given code? In training mode a random model is picked always
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

    random_prob = .2 #TODO: Adapt Exploration vs exploitation parameter
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80% walk in any direction. wait 20%. Bomb 0%
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0, .0])

    self.logger.debug("Querying model for action.")



    features = state_to_features(self, game_state) #get features from game_state
    self.logger.info(f"FEATURE CALCULATED")
    #return np.random.choice(ACTIONS, p=self.model[0])
    return ACTIONS[np.argmax(self.model[features])] #Gives action with maximal reward for given state


#The following two functions are defined in here so they can also be used in the train.py
def look_for_targets(free_space, start, targets, logger=None, dir=False):
    """Find distance to the closest target (target can be specified in use (coin/opponent/crate...))

    Performs a breadth-first search of the reachable free tiles until a special target is encountered.
    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        distance to the closest target or direction of closest target
    """
    if len(targets) == 0:
        return 17

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    if(dir):
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

        while True:
            if parent_dict[current] == start: return best_dist,current
            current = parent_dict[current]

    return best_dist

def destruction(arena, others_map, x,y):
    """
    Function that returns for a given tile, how much damage (crates/enemies) a bomb exploding at that time would create


    INPUT: (x,y) coordinates of tile
    OUTPUT: count of enemies/crates, that would be destroyed
    """

    possible_destruction=0
    for (i, j) in [(x + h, y) for h in range(-3, 4)] + [(x, y + h) for h in range(-3, 4)]:
        if (0 < i< arena.shape[0]) and (0 < j< arena.shape[1]):
            if arena[i,j]==1:
                possible_destruction+=1
            elif others_map[i,j]==1:
                possible_destruction+=1
    return possible_destruction



def state_to_features(self, game_state: dict) -> np.array:

    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    INPUT:
    game_state:  A dictionary describing the current game board.


    OUTPUT:
    tuple with 6 entries for the features
    """



    features = np.zeros(6, dtype='int')




    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Gather information about the game state
    arena = game_state['field']
    explosion_map = game_state['explosion_map']
    step = game_state['step']
    n,s,b,(x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [(n, s, b, xy) for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']

    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)                                             #Can be used as a measure for danger: 5 is undangerous, 0 is sure death

    bomb_positions = np.zeros(arena.shape) #1 where a bomb is placed, zero elsewhere
    for (xb, yb), t in bombs:
        bomb_positions[xb,yb] = 1
    others_map = np.zeros(arena.shape) #1 where another agent is placed, zero elsewhere
    for (n, s, b, xy) in others:
        others_map[xy[0],xy[1]] = 1

    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    walls = [(x, y) for x in cols for y in rows if (arena[x, y] == -1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    free_tiles = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)]
    free_space = arena == 0 #For the function



    #Phase Feature (5) (Needed for other features, because of that determined first)
    #Determine total number of found coins
    total_agents = 0 #TODO: Define depending on game mode, later always 3=opponent number)
    totalscore=0
    for n,s,b,(x,y) in others:
        totalscore+=s
    totalcoins=totalscore-((total_agents-len(others))*5)

    if(len(coins)>0):
        features[5]=0
    elif(totalcoins<9):
        features[5]=1
    else:
        features[5]=2


    #Neighbor tile features (0-3)
    #All the three are needed for the pase dependent decision
    distance_nextcoin = look_for_targets(free_space, (x, y), coins, self.logger) #distance to closest coin
    distance_nextopponent = look_for_targets(free_space,(x , y), [(xy) for (n, s, b, xy) in others], self.logger) #distance to closest neighbor
    distance_nextcrate = look_for_targets(free_space, (x, y), crates, self.logger) #distance to closest crate
    #how many crates/enemies would a bomb on this tile exploding right now destroy
    possible_destruction= destruction(arena, others_map, x,y)

    for num, (i,j) in enumerate([(x + h, y) for h in [-1, 1]] + [(x, y + h) for h in [-1, 1]]):
        #if tile is free
        if (arena[i,j] == 0):
            features[num]=0
        # if tile is Wall, crate, sure death, bomb, or other agent
        if (arena[i,j] == -1 or arena[i,j]== 1 or bomb_map[i,j]==1 or bomb_positions[i,j]==1 or others_map[i,j]==1 or explosion_map[i,j]>0):
            features[num]=1
        # possible danger (a bomb in the area may explode soon)
        elif (arena[i,j] == 0 and bomb_map[i,j]<=4):
            features[num]=2
        #Phase dependent value
        else:
            if(features[5]==0):
                if(look_for_targets(free_space, (i, j), coins, self.logger)<distance_nextcoin or (i,j) in coins):
                    features[num]=3
            elif(features[5]==1):
                if((destruction(arena, others_map,i,j)>possible_destruction) or (look_for_targets(free_space, (i, j), crates, self.logger)<distance_nextcrate)):
                    features[num]=3
            elif(features[5]==2):
                if((look_for_targets(free_space,(i , j), [(xy) for (n, s, b, xy) in others], self.logger)<distance_nextopponent) and bomb_map[i,j]>4):
                    features[num]=3



    #Feature for current tile (4)
    #wait does not lead to sure death and (can not place bomb (current field is bomb or bomb was placed recently) or placing bomb leads to sure death(trapped))
    if(bomb_map[x,y]>1 and (bomb_positions[x,y]==1 or features[0]==features[1]==features[2]==features[3]==1 or not b)):
        features[4]=0
    #not (directly) trapped and the possible destruction is (1 or 2)/(3-5)/(>6)
    elif(features[0]!=1 and features[1]!=1 and features[2]!= 1 and features[3]!=1 and (1 <= possible_destruction < 3)):
        features[4]=1
    elif(features[0]!=1 and features[1]!=1 and features[2]!= 1 and features[3]!=1 and (3 <= possible_destruction < 6)):
        features[4]=2
    elif(features[0]!=1 and features[1]!=1 and features[2]!= 1 and features[3]!=1 and (6 <= possible_destruction)):
        features[4]=3
    if(bomb_map[x,y]==1 or bomb_positions[x,y]==1): #Bomb placed on current tile or wait is safe death
        features[4]=4




    output = tuple(features)
    #self.logger.debug(f'Features Calculated: {output}')
    return output
