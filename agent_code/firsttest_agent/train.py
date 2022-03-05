from collections import namedtuple, deque


import numpy as np
import random
import pickle
from typing import List
import events as e
from .callbacks import state_to_features, look_for_targets, destruction

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
#RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
TEST = "TEST"
APPROACH_COIN = "APPROACH_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
VICTORY = "VICTORY"
REPEATED_MOVE = "REPEATED_MOVE"

#Make a dictionary to get position in vector out of action
action = {'UP': 0, 'RIGHT': 1, 'DOWN': 2,'LEFT': 3 , 'WAIT': 4, 'BOMB': 5}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    # Setup an array that will note transition tuples
    # (s, a, s', r)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.


    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    #Define hyperparameters for training
    alpha=0.1
    gamma=0.9
    
    if(len(self.transitions)>2):
        previous_actions = [a for (s,a,ns,r) in self.transitions]
        if(self_action == previous_actions[1] and previous_actions[0]==previous_actions[2]):
            events.append(REPEATED_MOVE)
    
    
    if(old_game_state != None and new_game_state != None):
        #Define old game state roperties TODO: Do not yet need all of them but probably for finetuning the rewardshaping
        old_arena = old_game_state['field']
        old_explosion_map = old_game_state['explosion_map']
        old_step = old_game_state['step']
        old_n,old_s,old_b,(old_x, old_y) = old_game_state['self']
        old_bombs = old_game_state['bombs']
        old_bomb_xys = [xy for (xy, t) in old_bombs]
        old_others = [(n, s, b, xy) for (n, s, b, xy) in old_game_state['others']]
        old_coins = old_game_state['coins']
        
        old_bomb_map = np.ones(old_arena.shape) * 5
        for (xb, yb), t in old_bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < old_bomb_map.shape[0]) and (0 < j < old_bomb_map.shape[1]):
                    old_bomb_map[i, j] = min(old_bomb_map[i, j], t)
        old_bomb_positions = np.zeros(old_arena.shape) #1 where a bomb is placed, zero elsewhere
        for (xb, yb), t in old_bombs:
            old_bomb_positions[xb,yb] = 1
        old_others_map = np.zeros(old_arena.shape) #1 where another agent is placed, zero elsewhere
        for (n, s, b, xy) in old_others:
            old_others_map[xy[0],xy[1]] = 1
        
        cols = range(1, old_arena.shape[0] - 1)
        rows = range(1, old_arena.shape[0] - 1)
        walls = [(x, y) for x in cols for y in rows if (old_arena[x, y] == -1)]
        old_crates = [(x, y) for x in cols for y in rows if (old_arena[x, y] == 1)]
        old_free_tiles = [(x, y) for x in cols for y in rows if (old_arena[x, y] == 0)]
        old_free_space = old_arena == 0
        
        #Define new game state properties TODO: Do not yet need all of them but probably for finetuning the rewardshaping
        new_arena = new_game_state['field']
        new_explosion_map = new_game_state['explosion_map']
        new_step = new_game_state['step']
        new_n,new_s,new_b,(new_x, new_y) = new_game_state['self']
        new_bombs = new_game_state['bombs']
        new_bomb_xys = [xy for (xy, t) in new_bombs]
        new_others = [(n, s, b, xy) for (n, s, b, xy) in new_game_state['others']]
        new_coins = new_game_state['coins']
        
        new_bomb_map = np.ones(new_arena.shape) * 5
        for (xb, yb), t in new_bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < new_bomb_map.shape[0]) and (0 < j < new_bomb_map.shape[1]):
                        new_bomb_map[i, j] = min(new_bomb_map[i, j], t)
                        
        new_bomb_positions = np.zeros(new_arena.shape) #1 where a bomb is placed, zero elsewhere
        for (xb, yb), t in new_bombs:
            new_bomb_positions[xb,yb] = 1
        new_others_map = np.zeros(new_arena.shape) #1 where another agent is placed, zero elsewhere
        for (n, s, b, xy) in new_others:
            new_others_map[xy[0],xy[1]] = 1
        
        new_crates = [(x, y) for x in cols for y in rows if (new_arena[x, y] == 1)]
        new_free_tiles = [(x, y) for x in cols for y in rows if (new_arena[x, y] == 0)]
        new_free_space = new_arena == 0 #For the function
        
        # Idea: Add your own events to hand out rewards
        if len(new_coins)>0 and look_for_targets(old_free_space, (old_x, old_y), old_coins)>look_for_targets(new_free_space, (new_x, new_y), new_coins):
            events.append(APPROACH_COIN)
        if len(new_coins)>0 and look_for_targets(old_free_space, (old_x, old_y), old_coins)<look_for_targets(new_free_space, (new_x, new_y), new_coins):
            events.append(AWAY_FROM_COIN)
    

        # state_to_features is defined in callbacks.py
        self.transitions.append(Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(self, events)))
        
        
        
        #Update Q-function(model) here according to lecture
        #TODO: Use SARSA instead of used Q-Learning
        self.model[state_to_features(self, old_game_state)][action[self_action]] = self.model[state_to_features(self, old_game_state)][action[self_action]] + alpha*(reward_from_events(self, events)+gamma*(np.max(self.model[state_to_features(self, new_game_state)]))-self.model[state_to_features(self, old_game_state)][action[self_action]]) #Q-Learning
        #self.model[state_to_features(self, old_game_state)][action[self_action]] = self.model[state_to_features(self, old_game_state)][action[self_action]] + alpha*(reward_from_events(self, events)+gamma*(np.max(self.model[state_to_features(self, new_game_state)]))-self.model[state_to_features(self, old_game_state)][action[self_action]]) #SARSA

        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
    
    


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))
    
    #TODO: Where should the reward for a victory be put into?
    score_others = [s for (n, s, b, xy) in last_game_state['others']]
    score_own = last_game_state['self'][1]
    if len(score_others)>0 and score_own>max(score_others):
        events.append(VICTORY)
    
    #Update Q-function(model) here
    #remove the last transition to get the one before the last
    self.transitions.pop()
    old_game_state= self.transitions.pop()[0]
    #self.model[state_to_features(self, old_game_state)] = self.model[state_to_features(self, old_game_state)] + alpha*(reward_from_events(self, events)+gamma*(np.max(self.model[state_to_features(self, last_game_state)]))-self.model[state_to_features(self, old_game_state)])


    # Store updated model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    #TODO: Create good rewards/penalties with good values. Avoid repeating moves somehow
    
    
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.INVALID_ACTION:-1,
        e.WAITED:-1,
        TEST: 0,
        APPROACH_COIN:1,
        AWAY_FROM_COIN:-2,
        VICTORY: 100,
        REPEATED_MOVE:-1,
        
        }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
