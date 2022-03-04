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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = np.ones((3840,6))*[.2, .2, .2, .2, .2, 0] #Initial guess #TODO: Replace 3840 by actual dimension of Q-table an (empty) working model (Q-Table)
        
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
    
    random_prob = .1 #TODO Adapt Exploration vs exploitation parameter
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80% walk in any direction. wait 10%. Bomb 10%
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])

    self.logger.debug("Querying model for action.")
    
    
    
    #state = features_to_state(state_to_features(game_state)) #get a number between 0 and the number of possible states from game_state
    return np.random.choice(ACTIONS, p=self.model[0]) #TODO: replace by feature (state) dependent move (Next line, once Q exists)
    #return = ACTIONS[np.argmax(Q[state])] #Gives action with maximal reward for state
    
    
def features_to_state(features):
    """
    INPUT:
    features: Array of feature values
    
    OUTPUT:
    state: int that encodes the current state in a number
    
    
    """
    
    #TODO: Build a function that gives a number between 0 and #{Possible STATES} from the given features
    return 0
    



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
    
    
    
    features = np.zeros(6)
    
    def look_for_targets(free_space, start, targets, logger=None):
        """Find direction of the closest target (coin)

        Performs a breadth-first search of the reachable free tiles until a target is encountered.
        Args:
            free_space: Boolean numpy array. True for free tiles and False for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
            logger: optional logger object for debugging.
        Returns:
            distance to the closest target
        """
        if len(targets) == 0:
            return None

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
        if logger: logger.debug(f'Suitable target found at {best}')
        # Determine the first step towards the best found target tile
        return best_dist
        
    
    
    
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    # Gather information about the game state
    arena = game_state['field']
    step = game_state['step']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [(n, s, b, xy) for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)                                             #Can be used as a measure for danger: 5 is undangerous, 0 is                                                                                    sure death
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
    
    
    distance_nextcoin = look_for_targets(free_space, (x, y), coins, self.logger) #distance to closest coin
    distance_nextneighbor = look_for_targets(free_space,(x , y), [(xy) for (n, s, b, xy) in others], self.logger) #distance to closest neighbor
    distance_nextcrate = look_for_targets(free_space, (x, y), crates, self.logger) #distance to closest crate
    
    
    
    #TODO: Neighbor tile features (0-3)
    for num, (i,j) in enumerate([(x + h, y) for h in [-1, 1]] + [(x, y + h) for h in [-1, 1]]):
        #if tile is free
        if (arena[i,j] == 0):
            features[num]=0
            
        # if tile is Wall, crate, sure death, bomb, or other agent
        if (arena[i,j] == -1 or arena[i,j]== 1 or bomb_map[i,j]==1 or bomb_positions[i,j]==1 or others_map[i,j]==1):
            features[num]=1
        # possible danger (a bomb in the area may explode soon)
        elif (arena[neighbor0[0],[neighbor0[1]]] == 0 and bomb_map[neighbor0[0],[neighbor0[1]]]<=4)
            features[num]=2
        #TODO: Design phase dependent value features[num]=3
        #elif ():
        #    features[num]=3
        
    
    #TODO: Design feature for current tile (4)
    #wait does not lead to sure death and (can not place bomb or placing bomb leads to sure death)
    if(bomb_map[x,y]>1 and (bomb_positions[x,y]==1 or )):
        features[4]=0
    
    if(bomb_map[x,y]==1 or bomb_positions[x,y]==1): #Bomb placed on current tile or wait is safe death
        features[3]=4
    
    
    #TODO: Phase Feature (5) FINISHED (determine first, when other features finished, move to top of feature function)
    #Determine total number of found coins
    if step==1:
        total_agents = len(others)
    totalscore=0
    for n,s,b,(x,y) in others:
        totalscore+=s
    totalcoins=totalscore-((total_agents-len(others))*5)

    
    
    if(len(coins)>0):
        features[5]=0
    elif(totalcoins<9):  #comment out for task 1
        features[5]=1
    else:
        features[5]=2
    
    #add test for checking push
    
    #return features
    return 0
