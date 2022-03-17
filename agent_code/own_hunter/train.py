import sys
import os
import numpy as np
import random
import pickle

from typing import List
from collections import namedtuple, deque

import events as e
from .callbacks import state_to_features, look_for_targets, destruction, potential_bomb, bombmap_and_freemap

sys.path.append(os.path.abspath(".."))

from agent_code.own_KGB import config


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'events'))


#Additional Structures
ACTION = {'UP': 0, 'RIGHT': 1, 'DOWN': 2,'LEFT': 3 , 'WAIT': 4, 'BOMB': 5}
ACTIONS_TO_INDEX    = {'UP': 0, 'RIGHT': 1, 'DOWN': 2,'LEFT': 3 , 'WAIT': 4, 'BOMB': 5}
ACTION_SYMMETRY     = {
    'HORIZONTAL': {0:2, 1:1, 2:0, 3:3, 4:4, 5:5},
    'VERTICAL': {0:0, 1:3, 2:2, 3:1, 4:4, 5:5},
    'ROTATION1': {0:1, 1:2, 2:3, 3:0, 4:4, 5:5},
    'POINT': {0:2, 1:3, 2:0, 3:1, 4:4, 5:5},
    'ROTATION3': {0:3, 1:0, 2:1, 3:2, 4:4, 5:5},
}

#Hyperparameters
TRANSITION_HISTORY_SIZE = 5  # keep only ... last transitions


# Events
REPEATED_MOVE = "REPEATED_MOVE"
DESTRUCTED_CRATE ="DESTRUCTED_CRATE"
CORRECT_WAIT = "CORRECT_WAIT"
WRONG_WAIT = "WRONG_WAIT"
STUPID_WAIT = "STUPID_WAIT"
CORRECT_DIRECTION = "CORRECT_DIRECTION"
GOOD_BOMB = "GOOD_BOMB"
UNNECCESSARY_BOMB = "UNNECCESSARY_BOMB"
BEST_BOMB = "BEST_BOMB"
BETTER_BOMB_POSSIBLE ="BETTER_BOMB_POSSIBLE"
SUICIDE_BOMB = "SUICIDE_BOMB"
SUICIDE_MOVE = "SUICIDE_MOVE"
STUPID_MOVE ="STUPID_MOVE"
WRONG_DIRECTION="WRONG_DIRECTION"
BOMB_CHANCE_MISSED ="BOMB_CHANCE_MISSED"





def setup_training(self):
    """
    Initialize self for training purpose.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if(self_action==None): #Must do this for training with rule_based_agent here, because wait is sometimes returned as None, which crashes the training
        self_action= 'WAIT'
        events.remove(e.INVALID_ACTION)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.logger.info(f"Action in game_events_occurred: {self_action}")


    #old_game_state is None in the first instance, so skip that one
    if(old_game_state == None):
        return

    #Define old game state roperties for reward shaping
    old_arena = old_game_state['field']
    old_n,old_s,old_b,(old_x, old_y) = old_game_state['self']
    old_bombs = old_game_state['bombs']
    old_free_space, old_bomb_map = bombmap_and_freemap(old_arena, old_bombs)
    old_features = self.features[-1]   #Features are sometimes ambigously and depend on BFS, that is why in rare cases the rewards to not match the action, but this is smoothed over time


    # Own events to hand out rewards
    if (old_features[0]==old_features[1]==old_features[2]==old_features[3]==1 and old_features[4]==0 and self_action=='WAIT'):
        events.append(CORRECT_WAIT)
    if ((old_features[0]==0 or old_features[0]==3 or old_features[1]==0 or old_features[1]==3 or old_features[2]==0 or old_features[2]==3 or old_features[3]==0 or old_features[3]==3) and self_action=='WAIT'):
        events.append(WRONG_WAIT)
    if (old_features[4]==4 and self_action =='WAIT' and (old_features[0]!=1 or old_features[1]!=1 or old_features[2]!=1 or old_features[3]!=1)):
        events.append(STUPID_WAIT)
    #MOVES: (APPROACH_COIN/AWAY_FROM_COIN is the same as this, thus not needed seperately)
    if (old_features[0]==3 and self_action=='LEFT'):
        events.append(CORRECT_DIRECTION)
    if (old_features[1]==3 and self_action=='RIGHT'):
        events.append(CORRECT_DIRECTION)
    if (old_features[2]==3 and self_action=='UP'):
        events.append(CORRECT_DIRECTION)
    if (old_features[3]==3 and self_action=='DOWN'):
        events.append(CORRECT_DIRECTION)


    #Only penalize if other move would really be bad and good chance not taken
    if (old_features[0]==3 and self_action!='LEFT' ):# and not (old_features[5]==2 and old_features[4]==2)):
        events.append(WRONG_DIRECTION)
    if (old_features[1]==3 and self_action!='RIGHT' ):#and not (old_features[5]==2 and old_features[4]==2)):
        events.append(WRONG_DIRECTION)
    if (old_features[2]==3 and self_action!='UP' ):#and not (old_features[5]==2 and old_features[4]==2)):
        events.append(WRONG_DIRECTION)
    if (old_features[3]==3 and self_action!='DOWN' ):#and not (old_features[5]==2 and old_features[4]==2)):
        events.append(WRONG_DIRECTION)
    if old_features[0]==1 and self_action =='LEFT':
        events.append(SUICIDE_MOVE)
    if old_features[1]==1 and self_action =='RIGHT':
        events.append(SUICIDE_MOVE)
    if old_features[2]==1 and self_action =='UP':
        events.append(SUICIDE_MOVE)
    if old_features[3]==1 and self_action =='DOWN':
        events.append(SUICIDE_MOVE)

    #BOMBS:
    if (old_features[4]==1 and self_action =='BOMB' ):
        events.append(UNNECCESSARY_BOMB)
    if (old_features[4]==2 and self_action =='BOMB' ):
        events.append(GOOD_BOMB)
    if (old_features[4]==3 and self_action =='BOMB'):
        events.append(BEST_BOMB)
    if (old_features[4]==0 and self_action =='BOMB'):
        events.append(SUICIDE_BOMB)

    if ((old_features[0]==3 or old_features[1]==3 or old_features[2]==3 or old_features[3]==3) and self_action =='BOMB' and old_bomb_map[old_x,old_y]==5 and (old_features[5]==1 or old_features[5]==2)):
        events.append(BETTER_BOMB_POSSIBLE)
    if ((old_features[0]!=3 and old_features[1]!=3 and old_features[2]!=3 and old_features[3]!=3) and (old_features[4]==2 or old_features[4]==3) and self_action!='BOMB'):
        events.append(BOMB_CHANCE_MISSED)


    # update score
    for (n, s, b, xy) in new_game_state['others']:
        self.score[n] = s


    self.transitions.append(Transition(old_game_state, self_action, new_game_state, events))

    #Update Q-function

    alpha = self.alpha
    gamma = self.gamma

    if config.TRULY_TRAIN:
        if (e.BOMB_EXPLODED in events): #If bomb exploded, rewards for the state where bomb was placed
            drop_state_features = self.features[0]
            self.logger.info(f"Features for drop state: {self.features[0]}")
            self.model[drop_state_features][5] = self.model[drop_state_features][5] + alpha*(reward_from_events(self, [ev for ev in events if (ev==e.CRATE_DESTROYED or ev==e.KILLED_OPPONENT  or ev==e.COIN_FOUND)])+gamma*(np.max(self.model[self.features[1]]))-self.model[drop_state_features][5]) #Only rewarding destruction, not the fact that a bomb exploded

        self.model[old_features][ACTION[self_action]] = self.model[old_features][ACTION[self_action]] + alpha*(reward_from_events(self, [ev for ev in events if (ev!=e.CRATE_DESTROYED and ev!=e.COIN_FOUND  and ev!=e.KILLED_OPPONENT) ])+gamma*(np.max(self.model[state_to_features(self, new_game_state)]))-self.model[old_features][ACTION[self_action]]) #Q-Learning

        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward_from_events(self, events)))




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
    if(last_action==None): #Must do this for training with rule_based_agent here, because wait is sometimes returned as None, which crashes the training
        last_action= 'WAIT'
        events.remove(e.INVALID_ACTION)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.logger.info(f"Action: {last_action}")

    # keep track of score
    for (n, s, b, xy) in last_game_state['others']:
        self.score[n] = s

    scores = np.array(list(self.score.values()))

    # log agents placement at end of the game
    score_own = last_game_state['self'][1]
    agent_placement = len(scores) - np.searchsorted(np.sort(scores), score_own) + 1
    self.logger.debug(f'Agents placement/score: {agent_placement},{score_own}')


    old_features = self.features[-1]
    alpha = self.alpha
    gamma = self.gamma


    if config.TRULY_TRAIN:
        if (e.BOMB_EXPLODED in events and e.KILLED_SELF not in events):
            drop_state_features = self.features[0]
            self.logger.info(f"Features for drop state: {self.features[0]}")
            self.model[drop_state_features][5] = self.model[drop_state_features][5] + alpha*(reward_from_events(self, [ev for ev in events if (ev==e.CRATE_DESTROYED or ev==e.KILLED_OPPONENT  or ev==e.COIN_FOUND)])+gamma*(np.max(self.model[self.features[1]]))-self.model[drop_state_features][5]) #Only rewarding destruction, not the fact that a bomb exploded
        #Q-Learning
        self.model[old_features][ACTION[last_action]] = self.model[old_features][ACTION[last_action]] + alpha*(reward_from_events(self, [ev for ev in events if (ev!=e.CRATE_DESTROYED and ev!=e.KILLED_OPPONENT and ev!=e.COIN_FOUND) ]))#Only this part because new state does not exist.  #Q-Learning

        # Store updated model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION:-5,
        e.KILLED_SELF:-5,
        e.CRATE_DESTROYED: 1,
        e.KILLED_OPPONENT: 5,
        e.COIN_FOUND: 1,
        CORRECT_WAIT:1,
        WRONG_WAIT:-1,
        CORRECT_DIRECTION:1,
        WRONG_DIRECTION:-2,
        BOMB_CHANCE_MISSED:-2,
        GOOD_BOMB:1,
        UNNECCESSARY_BOMB:-2,
        BETTER_BOMB_POSSIBLE:-2,
        BEST_BOMB:6,
        SUICIDE_BOMB:-5,
        SUICIDE_MOVE:-5,
        STUPID_WAIT:-5,
        e.MOVED_LEFT:-.25,
        e.MOVED_DOWN:-.25,
        e.MOVED_UP:-.25,
        e.MOVED_RIGHT:-.25,



        }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
