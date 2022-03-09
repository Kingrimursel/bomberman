import os
import pickle
import random
from collections import deque
#import settings as s
import numpy as np


#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def look_for_targets(free_space, start, targets, logger=None, dir=False):
    """Find distance to the closest target (target can be specified in use (coin/opponent/crate...))

    Performs a breadth-first search of the reachable free tiles until a special target is encountered.
    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        distance to the closest target or direction of closest target and Boolean for wether the target is reachable
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

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """


    self.logger.info("Loading model from saved state.")
    with open("my-saved-model.pt", "rb") as file:
        self.model = pickle.load(file)
    self.something = 78
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0




def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0




def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
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
    random.shuffle(action_ideas)

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
    if(len(targets)>0):
        d = look_for_targets(free_space, (x, y), targets, self.logger, dir=True)[1]
    else: d=None
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


#The following two functions are defined in here so they can also be used in the train.py


def destruction(arena, others, x,y):
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
            elif (i,j) in others:
                possible_destruction+=1
    return possible_destruction

def potential_bomb(arena,x,y):
    """
    Function that returns for a given tile (x,y), which tiles a bomb would hit, if dropped at (x,y)


    INPUT: (x,y) coordinates of tile
    OUTPUT: Array of all affected coordinates
    """

    possible_hit=[]
    for (i, j) in [(x + h, y) for h in range(-3, 4)] + [(x, y + h) for h in range(-3, 4)]:
        if (0 < i< arena.shape[0]) and (0 < j< arena.shape[1]):
            possible_hit = possible_hit + [(i,j)]
    return possible_hit



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
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)   #Can be used as a measure for danger: 5 is nothing, 0 is sure death
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    walls = [(x, y) for x in cols for y in rows if (arena[x, y] == -1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    free_tiles = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)]
    free_space = arena == 0 #For the function


    #All the three are needed for the phase dependent decision
    if(len(coins)>0):
        distance_nextcoin, closest_to_coin, coin_reachable = look_for_targets(free_space, (x, y), coins, self.logger, dir=True) #distance to closest coin
    else: coin_reachable = False #If no coin visible, set also to not reachable
    if(len(crates)>0):
        distance_nextcrate, closest_to_crate, _ = look_for_targets(free_space, (x, y), crates, self.logger, dir=True) #distance to closest crate
    if bomb_map[x,y]<5:#(np.count_nonzero(bomb_map < 5)>0 or np.count_nonzero(explosion_map == 1)>0):
        distance_nextsafe, closest_to_safe, safe_reachable = look_for_targets(free_space, (x, y), [(x,y) for (x,y) in np.array(np.where(bomb_map+explosion_map==5)).T if arena[x,y]==0], self.logger, dir=True) #distance to closest safe tile


    potential_bomb
    #Phase Feature (5) (Needed for other features, because of that determined first)
    #Determine total number of found coins
    total_agents = 0 #TODO: Define depending on game mode, later always 3=opponent number)
    totalscore=0
    for n,s,b,(xo,yo) in others:
        totalscore+=s
    totalcoins=totalscore-((total_agents-len(others))*5)

    if(len(coins)>0 and coin_reachable==True):
        features[5]=0
    elif(totalcoins<9):
        features[5]=1
    else:
        features[5]=2






    possible_destruction = destruction(arena, others,x, y)


    highest_destruction = max([destruction(arena, others, i,j) for (i,j) in [(x + h, y) for h in [-1, 1]] + [(x, y + h) for h in [-1, 1]]])
    possible_bomb = potential_bomb(arena, x, y)
    danger_tiles = possible_bomb + [(i,j) for (i,j) in free_tiles if (bomb_map+explosion_map)[i,j]!=5] #All tiles affected by any explosion
    potential_escape = [(i,j) for (i,j) in free_tiles if (i,j) not in danger_tiles] #All tiles, that would still be safe when bomb is dropped
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

        if(len(coins)>0 and features[5]==0):
            #When tile is closest to next coin, mark with this value
            if((i,j)==closest_to_coin and (bomb_map+explosion_map)[i,j]==5):
                features[num]=3
        #If in danger, set way out to 3, if not in danger set tile with most destruction potential to 3 or set way to next crate to 3, if no destruction possible
        elif(features[5]==1):
            if(bomb_map[x,y]<5 and (i,j)==closest_to_safe):
                features[num]=3
            #Encode neighbor tile with most destruction potential with 3
            elif(bomb_map[x,y]==5  and destruction(arena, others,i, j)==highest_destruction):
                features[num]=3
            #If no destruction possible, mark way to next crate with 3.
            if(len(crates)>0 and bomb_map[x,y]==5 and highest_destruction==0 and (i,j)==closest_to_crate and distance_nextcrate !=1):
                features[num]=3
        #The following case will not occur in this section (only with opponents, later tasks)
        elif(features[5]==2):
            features[num]=3
        # if tile is Wall, crate, bomb, or death (bomb will explode or explosion lasts for next step) stringer than everything else, so calculated last
        if (arena[i,j] == -1 or arena[i,j] == 1 or (i,j) in bomb_xys or bomb_map[i,j]==0 or explosion_map[i,j]==1):
            features[num]=1




    #Feature for current tile (4) (in coin task always 1)
    #wait does not lead to sure death and (can not place bomb or moving leads to sure death or bomb placement on current tile would be safe death(No escape tile reachable))
    if(bomb_map[x,y]>0 and (not b or features[0]==features[1]==features[2]==features[3]==1)or not look_for_targets(free_space, (x, y), potential_escape , self.logger, dir=True)[2]):
        features[4]=0
    #not (directly) trapped and the possible destruction is (0)/(1-5)/(>6)
    elif((features[0]!=1 or features[1]!=1 or features[2]!= 1 or features[3]!=1) and (possible_destruction == 0)):
        features[4]=1
    elif((features[0]!=1 or features[1]!=1 or features[2]!= 1 or features[3]!=1) and (1 <= possible_destruction < 6)):
        features[4]=2
    elif((features[0]!=1 or features[1]!=1 or features[2]!= 1 or features[3]!=1) and (6 <= possible_destruction)):
        features[4]=3
    if(bomb_map[x,y]==0 or (x,y) in bomb_xys ): #Bomb is placed on current tile, or wait is safe death
        features[4]=4
    if(features[5]==0 and distance_nextcoin<distance_nextcrate): #Only stop dropping bombs, when coin is close
        features[4]=1



    output = tuple(features)
    #self.logger.debug(f'Features Calculated: {output}')
    return output
