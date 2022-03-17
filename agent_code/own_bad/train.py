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
    self.alpha   = config.ALPHA
    self.gamma   = config.GAMMA
    self.epsilon = config.EPSILON

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def get_equivalent_features(feature: tuple):
    """
    Using symmetries of the playground to get equivalent feature of the passed feature.

    :param feature: A tuple of the indices in feature space/self.model except the last axis
        which indicates the action to take in this state. Remember that the feature indeices are
        build up like this: (left, right, above, below, current, game mode).
    :return: Three equivalent features in the follwoing order:
        (horizontal reflection, vertical reflectino, point reflection)
    """
    f_left    = feature[0]
    f_right   = feature[1]
    f_above   = feature[2]
    f_below   = feature[3]
    f_current = feature[4]
    f_phase   = feature[5]

    # horizontal reflection
    feature_h = tuple([f_left, f_right, f_below, f_above, f_current, f_phase])
    # vertiacl reflection
    feature_v = tuple([f_right, f_left, f_above, f_below, f_current, f_phase])
    #Rotation by pi/2
    feature_r1 = tuple([f_below, f_above, f_left, f_right, f_current, f_phase])
    # point reflecion/ rotation by pi
    feature_r2 = tuple([f_right, f_left, f_below, f_above, f_current, f_phase])
    #Rotation by 3pi/2
    feature_r3 = tuple([f_above, f_below, f_right, f_left, f_current, f_phase])

    return [feature, feature_h, feature_v, feature_r1, feature_r2, feature_r3] #TODO: Only use equivalent features if they are not the same later

def get_equivalent_actions(action: int):
    """
    Using symmetries of the playground to get equivalent feature of the passed feature.

    :param action: The index of the last axis in feature space/self.model.
    :return: Five actions for equivalent features in the following order:
        (horizontal reflection, vertical reflection, rotation by pi/2, point reflection, rotation by 3pi/2)
    """
    # horizontal reflection
    action_h  = ACTION_SYMMETRY['HORIZONTAL'][action]
    # vertiacl reflection
    action_v  = ACTION_SYMMETRY['VERTICAL'][action]
    #Pi/2 Rotation
    action_r1  = ACTION_SYMMETRY['ROTATION1'][action]
    # point reflection and Pi rotation
    action_r2  = ACTION_SYMMETRY['POINT'][action]
    #3Pi/2 Rotation
    action_r3  = ACTION_SYMMETRY['ROTATION3'][action]

    return [action, action_h, action_v, action_r1, action_r2 ,action_r3]

def update_Q(self, nstep=False, SARSA=False, Qlearning=True, bomb_exploded=False, last_move=False):
    """
    Performs update on Q depending on which method was chosen. By default normal Q-learning is chosen.

    :param self: This object is passed to all callbacks.
    :param nstep/SARSA/Qlearning: Boolean which indicates which update method should be chosen.
    :param bomb_exploded: Boolean which indicates if a bomb exploded, to update the move where bomb was dropped
    """
    if np.sum([nstep, Qlearning, SARSA]) != 1:
        raise TypeError(f"Choose only and at least one learning method. {np.sum([nstep, Qlearning, SARSA])} were given.")

    #TODO: Adapt n-step and SARSA to new symmetries
    if nstep:
        transition_0 = self.transitions[0]
        n            = len(self.transitions)

        if transition_0[0]['step']<TRANSITION_HISTORY_SIZE or n<1:
            return

        transition_n = self.transitions[-1]
        self.transitions.append(transition_n)

        feature_0 = get_equivalent_features(state_to_features(self, transition_0[0]))
        action_0  = get_equivalent_actions(ACTIONS_TO_INDEX[transition_0[1]])
        feature_n = get_equivalent_features(state_to_features(self, transition_n[0]))
        action_n  = get_equivalent_actions(ACTIONS_TO_INDEX[transition_n[1]])

        prev_rewards = np.array([r for (_, _, _, r) in self.transitions])
        gamma_pow    = np.power(np.ones(n) * self.gamma, np.arange(n))

        for i in range(4):
            self.model[feature_0[i]][action_0[i]] = self.model[feature_0[i]][action_0[i]] + self.alpha * np.sum(gamma_pow*prev_rewards + self.gamma**n * np.max(self.model[feature_n[i]]) - self.model[feature_0[i]][action_0[i]])

    elif SARSA:
        if len(self.transitions) < 2:
            return
        if bomb_exploded: #If a bomb exploded, use this funtion to reward the bomb drop move
            transition = self.transitions[0]
            old_feature     = get_equivalent_features(self.features[0])
            new_feature     = get_equivalent_features(self.features[1])
            old_action      = get_equivalent_actions(ACTIONS_TO_INDEX[transition[1]])
            new_action      = get_equivalent_actions(self.transitions[1][1])
            rewards         = reward_from_events(self, [ev for ev in self.transitions[-1][3] if (ev==e.CRATE_DESTROYED or ev==e.KILLED_OPPONENT  or ev==e.COIN_FOUND)])


            self.logger.debug(f'Equivalent features: {old_feature}')
            self.logger.debug(f'action: {old_action}')
            self.logger.debug(f'transition action: {ACTIONS_TO_INDEX[transition[1]]}')


            for i in np.unique(old_feature, return_index=True)[1]:
                self.model[old_feature[i]][old_action[i]] = self.model[old_feature[i]][old_action[i]] + self.alpha*(rewards + self.gamma*self.model[new_feature[i]][new_action[i]] - self.model[old_feature[i]][old_action[i]])
            return


        if last_move: #there is no new state in last move, so define some new way of Q-learning to reward e.g. suicide moves
            transition      = self.transitions[-1]
            old_feature     = get_equivalent_features(self.features[-1])
            old_action      = get_equivalent_actions(ACTIONS_TO_INDEX[transition[1]])
            rewards         = reward_from_events(self, [ev for ev in transition[3] if (ev!=e.CRATE_DESTROYED or ev!=e.KILLED_OPPONENT  or ev!=e.COIN_FOUND)])


            self.logger.debug(f'Equivalent features: {old_feature}')
            self.logger.debug(f'action: {old_action}')
            self.logger.debug(f'transition action: {ACTIONS_TO_INDEX[transition[1]]}')


            for i in np.unique(old_feature, return_index=True)[1]:
                self.model[old_feature[i]][old_action[i]] = self.model[old_feature[i]][old_action[i]] + self.alpha*(rewards)
            return

        transition  = self.transitions[-2]
        old_feature = get_equivalent_features(self.features[-2])
        new_feature = get_equivalent_features(self.features[-1])
        old_action  = get_equivalent_actions(ACTIONS_TO_INDEX[transition[1]])
        new_action  = get_equivalent_actions(self.transitions[-1][1])
        rewards     = reward_from_events(self, [ev for ev in transition[3] if (ev!=e.CRATE_DESTROYED and ev!=e.COIN_FOUND and ev!= e.KILLED_OPPONENT)])

        self.logger.debug(f'Equivalent features: {old_feature}')
        self.logger.debug(f'action: {old_action}')
        self.logger.debug(f'transition action: {ACTIONS_TO_INDEX[transition[1]]}')


        for i in np.unique(old_feature, return_index=True)[1]:
            self.model[old_feature[i]][old_action[i]] = self.model[old_feature[i]][old_action[i]] + self.alpha*(rewards + self.gamma*self.model[new_feature[i]][new_action[i]] - self.model[old_feature[i]][old_action[i]])


    elif Qlearning:
        if len(self.transitions) < 2:
            return
        if bomb_exploded: #If a bomb exploded, use this funtion to reward the bomb drop move
            transition = self.transitions[0]
            old_feature     = get_equivalent_features(self.features[0])
            new_feature     = get_equivalent_features(self.features[1])
            old_action      = get_equivalent_actions(ACTIONS_TO_INDEX[transition[1]])
            new_action      = get_equivalent_actions(np.argmax(self.model[self.features[1]]))
            rewards         = reward_from_events(self, [ev for ev in self.transitions[-1][3] if (ev==e.CRATE_DESTROYED or ev==e.KILLED_OPPONENT  or ev==e.COIN_FOUND)])

            self.logger.debug(f'Rewards: {rewards}')
            self.logger.debug(f'Equivalent features in bomb drop state: {old_feature}')
            self.logger.debug(f'action: {old_action}')
            self.logger.debug(f'transition action: {transition[1]}')


            for i in np.unique(old_feature, return_index=True)[1]:
                self.model[old_feature[i]][old_action[i]] = self.model[old_feature[i]][old_action[i]] + self.alpha*(rewards + self.gamma*self.model[new_feature[i]][new_action[i]] - self.model[old_feature[i]][old_action[i]])
            return


        if last_move: #there is no new state in last move, so define some new way of Q-learning to reward e.g. suicide moves
            transition      = self.transitions[-1]
            old_feature     = get_equivalent_features(self.features[-1])
            old_action      = get_equivalent_actions(ACTIONS_TO_INDEX[transition[1]])
            rewards         = reward_from_events(self, [ev for ev in transition[3] if (ev!=e.CRATE_DESTROYED and ev!=e.KILLED_OPPONENT and ev!=e.COIN_FOUND)])


            self.logger.debug(f'Equivalent  last features: {old_feature}')
            self.logger.debug(f'last action: {old_action}')
            self.logger.debug(f'last transition action: {transition[1]}')


            for i in np.unique(old_feature, return_index=True)[1]:
                self.model[old_feature[i]][old_action[i]] = self.model[old_feature[i]][old_action[i]] + self.alpha*(rewards)
            return

        transition  = self.transitions[-2]
        old_feature = get_equivalent_features(self.features[-2])
        new_feature = get_equivalent_features(self.features[-1])
        old_action  = get_equivalent_actions(ACTIONS_TO_INDEX[transition[1]])
        new_action  = get_equivalent_actions(np.argmax(self.model[self.features[-1]]))
        rewards     = reward_from_events(self, [ev for ev in transition[3] if (ev!=e.CRATE_DESTROYED and ev!=e.COIN_FOUND and ev!= e.KILLED_OPPONENT)])

        self.logger.debug(f'Equivalent features: {old_feature}')
        self.logger.debug(f'action: {old_action}')
        self.logger.debug(f'transition action: {transition[1]}')


        for i in np.unique(old_feature, return_index=True)[1]:
            print(old_feature[i])
            self.model[old_feature[i]][old_action[i]] = self.model[old_feature[i]][old_action[i]] + self.alpha*(rewards + self.gamma*self.model[new_feature[i]][new_action[i]] - self.model[old_feature[i]][old_action[i]])





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
            equivalent_features = get_equivalent_features(drop_state_features) #Get all equivalent features to bomb_drop_state
            equivalent_new_features = get_equivalent_features(self.features[1])
            for i in np.unique(equivalent_features, return_index=True)[1]:
                self.logger.info(f"Update for: {equivalent_features[i]}")
                self.model[equivalent_features[i]][5] = self.model[equivalent_features[i]][5] + alpha*(reward_from_events(self, [ev for ev in events if (ev==e.CRATE_DESTROYED or ev==e.KILLED_OPPONENT  or ev==e.COIN_FOUND)])+gamma*(np.max(self.model[equivalent_new_features[i]]))-self.model[equivalent_features[i]][5]) #Only rewarding destruction, not the fact that a bomb exploded
        equivalent_features = get_equivalent_features(old_features) #Get all equivalent features to latest feature
        equivalent_new_features = get_equivalent_features(state_to_features(self, new_game_state))
        equivalent_actions = get_equivalent_actions(ACTIONS_TO_INDEX[self_action])
        for i in np.unique(equivalent_features, return_index=True)[1]:
            self.logger.info(f"Update for: {equivalent_features[i]}")
            self.model[equivalent_features[i]][equivalent_actions[i]] = self.model[equivalent_features[i]][equivalent_actions[i]] + alpha*(reward_from_events(self, [ev for ev in events if (ev!=e.CRATE_DESTROYED and ev!=e.COIN_FOUND  and ev!=e.KILLED_OPPONENT) ])+gamma*(np.max(self.model[equivalent_new_features[i]]))-self.model[equivalent_features[i]][equivalent_actions[i]]) #Q-Learning

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
