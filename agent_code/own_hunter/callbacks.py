import os
import pickle
import random

import numpy as np

from collections import  deque

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

    self.features= deque(maxlen=5)

def act(self, game_state: dict) -> str:
    """
    Agent parses the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    random_prob=.01 #random move in 20, 10, 5, 2, 1% of cases

    features = state_to_features(self, game_state) #get features from game_state
    self.features.append(features)
    self.logger.info(f"FEATURES CALCULATED")
    self.logger.info(f"Features: {features}")

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 0% walk in any direction. wait 0%. Bomb 100%
        #return np.random.choice(ACTIONS, p=[.0, .0, .0, .0, .0, 1])
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .0, .2])

    self.logger.debug("Querying model for action.")
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
            distance to the closest target
        OR
            if dir==True:  distance to closest target, direction of closest target and Boolean for wether the target is reachable
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
    #if logger: logger.debug(f'Suitable target found with distance {best_dist}')
    # Determine the first step towards the best found target tile
    current = best

    while True and dir:
        if parent_dict[current] == start:
            if d==0:
                return best_dist,current,True
            else: #Return if a path to the coin exists
                return best_dist,current,False
        current = parent_dict[current]

    return best_dist



def potential_bomb(arena,x,y):
    """
    Function that returns for a given tile (x,y), which tiles a bomb would hit, if dropped at (x,y)


    INPUT: (x,y) coordinates of tile
    OUTPUT: Array of all affected coordinates
    """
    array=[]
    for h in range(1, 4):
        if(arena[x-h,y]==-1): break
        array.append((x - h, y))
    for h in range(0, 4):
        if(arena[x+h,y]==-1): break
        array.append((x + h, y))
    for h in range(1, 4):
        if(arena[x,y-h]==-1): break
        array.append((x, y - h))
    for h in range(0, 4):
        if(arena[x,y+h]==-1): break
        array.append((x, y + h))
    return array

def destruction(arena, others, x,y):
    """
    Function that returns for a given tile, how much damage (crates/enemies) a bomb exploding at that time would create


    INPUT: (x,y) coordinates of tile
    OUTPUT: count of enemies/crates, that would be destroyed
    """
    possible_destruction=0

    array = potential_bomb(arena, x, y)
    for (i, j) in array:
        if arena[i,j]==1:
            possible_destruction+=1
        elif (i,j) in others:
            possible_destruction+=1
    return possible_destruction

def bombmap_and_freemap(arena, bombs):
    bomb_map = np.ones(arena.shape) * 5
    free_space = arena == 0 #For the function
    for (xb, yb), t in bombs:
        free_space[xb,yb]=False #Positions where bombs are are also non passable
        for (i, j) in potential_bomb(arena, xb, yb):
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)   #Can be used as a measure for danger: 5 is nothing, 0 is sure death
    return free_space, bomb_map

def state_to_features(self, game_state: dict) -> np.array:

    """
    Converts the game state to the input of the model, i.e.
    a feature vector.


    INPUT:
    self: instance on which it is used (agent)
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
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    n,s,b,(x, y) = game_state['self']
    others = [(n, s, b, xy) for (n, s, b, xy) in game_state['others']] #For calculating the number of coins collected yet
    coins = game_state['coins']
    free_space, bomb_map = bombmap_and_freemap(arena, bombs)
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    walls = [(x, y) for x in cols for y in rows if (arena[x, y] == -1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    free_tiles = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)]
    for (xb, yb) in bomb_xys:
        free_tiles.remove((xb,yb))
    others_free_tiles = free_tiles.copy()
    if (x,y) in others_free_tiles: others_free_tiles.remove((x,y))
    others_free_space = free_space.copy()
    others_free_space[x,y]=False
    others_coord = []
    others_closest_to_safe =[]
    for no,so,bo,(xo,yo) in others:
        free_space[xo,yo]=False
        if((xo,yo) in free_tiles): free_tiles.remove((xo,yo)) #If bomb dropped, then the point is not in list anymore
        others_coord.append((xo,yo))
        if bomb_map[xo,yo]<5: #Calculate for every opponent the way out of a bomb radius if they are in
            _, others_closest, _ = look_for_targets(free_space, (xo, yo), [(x,y) for (x,y) in np.array(np.where(bomb_map+explosion_map==5)).T if free_space[x,y]], self.logger, dir=True) #distance to closest safe tile
            others_closest_to_safe.append(others_closest)


    #All four are needed for the phase dependent decision
    if(len(coins)>0):
        distance_nextcoin, closest_to_coin, coin_reachable = look_for_targets(free_space, (x, y), coins, self.logger, dir=True) #distance to closest coin
    else: coin_reachable = False #If no coin visible, set also to not reachable
    if(len(crates)>0):
        distance_nextcrate, closest_to_crate, _ = look_for_targets(free_space, (x, y), crates, self.logger, dir=True) #distance to closest crate
    if bomb_map[x,y]<5:#(np.count_nonzero(bomb_map < 5)>0 or np.count_nonzero(explosion_map == 1)>0):
        distance_nextsafe, closest_to_safe, safe_reachable = look_for_targets(free_space, (x, y), [(x,y) for (x,y) in np.array(np.where(bomb_map+explosion_map==5)).T if free_space[x,y]], self.logger, dir=True) #distance to closest safe tile
    if len(others)>0:
        distance_nextopponent, closest_to_opponent, opponent_reachable = look_for_targets(free_space, (x,y), others_coord, self.logger, dir=True)

    #Phase Feature (5) (Needed for other features, because of that determined first)
    if(len(coins)>0 and coin_reachable==True and (len(crates)==0 or distance_nextcoin<distance_nextcrate+5)): #Only change to collect mode if coin is in a reasonable region
        features[5]=0
    elif(len(coins)>0 and len(crates)>0):
        features[5]=1
    else:
        features[5]=2

    #Possible destruction for bomb placed at current tile
    possible_destruction = destruction(arena, others_coord,x, y)
    #Highest destruction for neighbor tiles
    highest_destruction = max([destruction(arena, others, i,j) for (i,j) in [(x + h, y) for h in [-1, 1]] + [(x, y + h) for h in [-1, 1]]]) #Highest destruction for all 4 neighbor tiles
    #All the neighbor tiles that lead to that highest value of destruction
    max_destruction_tiles = ([(x + h, y) for h in [-1, 1] if destruction(arena,others, x+h, y)==highest_destruction] + [(x, y + h) for h in [-1, 1] if destruction(arena,others, x, y+h)==highest_destruction]) #Pick Neighbor Tiles with highest destruction
    max_destruction_tile = max_destruction_tiles[0]
    #All tiles affected by a potential bomb drop at current tile
    possible_bomb = potential_bomb(arena, x, y)
    #All tiles with potential danger, if bomb was dropped
    danger_tiles = possible_bomb + [(i,j) for (i,j) in free_tiles if (bomb_map+explosion_map)[i,j]<5] #All tiles affected by any other explosion in move after potential bomb drop
    others_danger_tiles = possible_bomb + [(i,j) for (i,j) in others_free_tiles if (bomb_map+explosion_map)[i,j]<5]
    potential_escape = [(i,j) for (i,j) in free_tiles if (i,j) not in danger_tiles] #All tiles, that would still be safe when bomb is dropped
    others_potential_escape = [(i,j) for (i,j) in others_free_tiles if (i,j) not in others_danger_tiles]

    #TODO: Implement feature that encourages bomb drop when opponent is trapped
    for (xo,yo) in others_coord:
        not_trapped = True
        if bomb_map[xo,yo]==5: #Calculate for every opponent if they are trapped by a potential bomb
            not_trapped = look_for_targets(others_free_space, (xo, yo), others_potential_escape, self.logger, dir=True)[2]
            if not not_trapped:
                self.logger.info(f"Is trapped")
                break



    #Neighbor tile features (0-3)
    #This is in order (Left, Right, Above, Below)
    for num, (i,j) in enumerate([(x + h, y) for h in [-1, 1]] + [(x, y + h) for h in [-1, 1]]):
        #if tile is free
        if (arena[i,j] == 0):
            features[num]=0
        #Phase dependent value
        # possible danger (a bomb reaching this tile explodes soon)
        if (bomb_map[i,j]<5):
            features[num]=2
        #If agent must leave current tile to survive, mark way to safe with 3
        if(bomb_map[x,y]<5 and distance_nextsafe==(bomb_map[x,y]+1) and (i,j)==closest_to_safe):
            features[num]=3

        if(len(coins)>0 and features[5]==0):
            #When tile is closest to next coin, mark with this value
            if((i,j)==closest_to_coin and (bomb_map+explosion_map)[i,j]==5 and ( (len(crates)>0 and distance_nextcoin<(distance_nextcrate+5)) or len(crates)==0 or not b)):
                features[num]=3

        #If in danger, set way out to 3, if not in danger set tile with most destruction potential to 3 or set way to next crate to 3, if no destruction possible
        elif(features[5]==1):
            #Encode neighbor tile with most destruction potential with 3
            if bomb_map[x,y]==5 and highest_destruction!=0:
                #if current tile has lower destruction potential
                if((i,j)==max_destruction_tile and highest_destruction>possible_destruction):
                    features[num]=3
                #no escape from current tile possible if bomb dropped and neighbor tile has (second) highest destruction potential
                elif(not look_for_targets(free_space, (x, y), potential_escape , self.logger, dir=True)[2] and (i,j)==max_destruction_tile):
                    features[num]=3
            #If no destruction possible, mark way to next crate with 3.
            elif(len(crates)>0 and bomb_map[x,y]==5 and highest_destruction==0 and (i,j)==closest_to_crate and distance_nextcrate !=1):
                features[num]=3

        #Neighbor features for last phase
        elif(features[5]==2):
            #Set 3 to the tile that is closest to opponent (The used look_for_targets function decides, wether an escape is neccessary(not neccessary == tile is not safe death)
            if(0<i<16 and 0<j<16 and look_for_targets(free_space, (i, j), [(x,y) for (x,y) in np.array(np.where(bomb_map+explosion_map==5)).T if free_space[x,y]], self.logger, dir=True)[0]<=(bomb_map[i,j]+1) and len(others)>0 and (i,j)==closest_to_opponent):
                if bomb_map[x,y]==5 or distance_nextsafe!=(bomb_map[x,y]+1):
                    features[num]=3
            if((i,j) in others_closest_to_safe and look_for_targets(free_space, (i, j), [(x,y) for (x,y) in np.array(np.where(bomb_map+explosion_map==5)).T if free_space[x,y]], self.logger, dir=True)[0]<=(bomb_map[i,j])): #If tile is on way to safe for opponent, but going there is no safe death ,set tile 3. (It is without the +1 at the and here, because that is the tile to go to, so when going there, the bomb timer counts one down, and then escape must still be possible)
                features[num]=3
            if(not not_trapped or (x,y) in others_closest_to_safe) and (bomb_map[x,y]==5 or distance_nextsafe<(bomb_map[x,y]+1)):
                features[num]=1

        #General for every game phase:
        if(len(others)==0 and len(crates)==0 and len(coins)==0): #Force to wait when everyone is dead and nothing to collect
            features[num]=1
        # if tile is other agent, Wall, crate, bomb, or death (bomb will explode or explosion lasts for next step or no escape from tile possible) stronger than everything else, so calculated last
        if ((i,j) in others_coord or arena[i,j] == -1 or arena[i,j] == 1 or (i,j) in bomb_xys or bomb_map[i,j]==0 or explosion_map[i,j]==1 or look_for_targets(free_space, (i, j), [(x,y) for (x,y) in np.array(np.where(bomb_map+explosion_map==5)).T if free_space[x,y]], self.logger, dir=True)[0]>=(bomb_map[i,j]+1) or not look_for_targets(free_space, (i, j), [(x,y) for (x,y) in np.array(np.where(bomb_map+explosion_map==5)).T if free_space[x,y]], self.logger, dir=True)[2]):
            features[num]=1




    #Feature for current tile (4) (in coin task always 1)
    #wait does not lead to sure death and (can not place bomb or moving leads to sure death or bomb placement on current tile would be safe death(No escape tile reachable))
    if(bomb_map[x,y]>0 and (not b or not look_for_targets(free_space, (x, y), potential_escape , self.logger, dir=True)[2])): #Removed: or features[0]==features[1]==features[2]==features[3]==1) TODO: is this smart?
        features[4]=0
    #not trapped and the possible destruction is (0)/(3-5)/(>6)
    elif(look_for_targets(free_space, (x, y), potential_escape , self.logger, dir=True)[2] and (possible_destruction == 0)):
        features[4]=1
    elif(look_for_targets(free_space, (x, y), potential_escape , self.logger, dir=True)[2] and (1 <= possible_destruction < 5)):
        features[4]=2
    elif(look_for_targets(free_space, (x, y), potential_escape , self.logger, dir=True)[2] and (5 <= possible_destruction)):
        features[4]=3
    #Bomb is placed on current tile, or wait is safe death
    if(bomb_map[x,y]==0 or (x,y) in bomb_xys or look_for_targets(free_space, (x, y), [(x,y) for (x,y) in np.array(np.where(bomb_map+explosion_map==5)).T if free_space[x,y]], self.logger, dir=True)[0]>=(bomb_map[x,y]+1)):
        features[4]=4
    if(features[5]==0 and len(coins)>0 and len(crates)>0 and distance_nextcoin<distance_nextcrate): #Only stop dropping bombs, when coin is close
        features[4]=1



    output = tuple(features)
    #self.logger.debug(f'Features Calculated: {output}')
    return output
