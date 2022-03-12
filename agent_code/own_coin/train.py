from collections import namedtuple, deque


import numpy as np
import random
import pickle
from typing import List
import events as e
from .callbacks import state_to_features, look_for_targets

from . import config

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
#RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
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

    # Setup an array that will note transition tuples, as they are defined above
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



    if(old_game_state != None and new_game_state != None):
        #Define old game state roperties TODO: Do not yet need all of them but probably for finetuning the rewardshaping
        old_arena = old_game_state['field']
        old_step = old_game_state['step']
        old_n,old_s,old_b,(old_x, old_y) = old_game_state['self']
        old_coins = old_game_state['coins']
        cols = range(1, old_arena.shape[0] - 1)
        rows = range(1, old_arena.shape[0] - 1)
        walls = [(x, y) for x in cols for y in rows if (old_arena[x, y] == -1)]
        old_free_tiles = [(x, y) for x in cols for y in rows if (old_arena[x, y] == 0)]
        old_free_space = old_arena == 0

        #Define new game state properties TODO: Do not yet need all of them but probably for finetuning the rewardshaping
        new_arena = new_game_state['field']
        new_step = new_game_state['step']
        new_n,new_s,new_b,(new_x, new_y) = new_game_state['self']
        new_coins = new_game_state['coins']
        new_free_tiles = [(x, y) for x in cols for y in rows if (new_arena[x, y] == 0)]
        new_free_space = new_arena == 0 #For the function

        # Own events to hand out rewards
        if len(new_coins)>0 and look_for_targets(old_free_space, (old_x, old_y), old_coins,dir=True)[1]==(new_x, new_y):
            events.append(APPROACH_COIN)
        if len(new_coins)>0 and look_for_targets(old_free_space, (old_x, old_y), old_coins,dir=True)[1]!=(new_x, new_y):
            events.append(AWAY_FROM_COIN)


        if(len(self.transitions)>=TRANSITION_HISTORY_SIZE):
            previous_actions = [a for (s,a,ns,r) in self.transitions]
            if(self_action == previous_actions[1] and previous_actions[0]==previous_actions[2] and previous_actions[1]!=previous_actions[2]):
                events.append(REPEATED_MOVE)

        #N-Step Q Learning:
        #     current_reward = reward_from_events(self, events)
        #
        #     old_states = np.array([old_state for (old_state,action,new_state,reward) in self.transitions])
        #     actions = np.array([action[a] for (old_state,a,new_state,reward) in self.transitions])
        #     new_states = np.array([new_state for (old_state,action,new_state,reward) in self.transitions])
        #     prev_rewards = np.array([reward for (old_state,action,new_state,reward) in self.transitions])
        #
        #     old_states = np.append(old_states, old_game_state)
        #     actions = np.append(actions, action[self_action])
        #     new_states = np.append(new_states, new_game_state)
        #     prev_rewards = np.append(prev_rewards, current_reward)
        #
        #     gamma_exp = np.array([self.gamma**k for k in range(0,len(actions))])
        #
        #
        #
        #     self.model[state_to_features(self, old_states[0])][actions[0]] = self.model[state_to_features(self, old_states[0])][actions[0]] + self.alpha*(np.sum(self.gamma_exp*prev_rewards +(self.gamma**TRANSITION_HISTORY_SIZE)*np.max(self.model[state_to_features(self, new_game_state)]))-self.model[state_to_features(self, old_states[0])][actions[0]])
        #     self.logger.info(f"Model updated")

        #Update Q-function(model) here
        #Q-Learning:
        self.model[state_to_features(self, old_game_state)][action[self_action]] = self.model[state_to_features(self, old_game_state)][action[self_action]] + self.alpha*(reward_from_events(self, events)+self.gamma*(np.max(self.model[state_to_features(self, new_game_state)]))-self.model[state_to_features(self, old_game_state)][action[self_action]]) #Q-Learning
        #SARSA:
        #self.model[state_to_features(self, old_game_state)][action[self_action]] = self.model[state_to_features(self, old_game_state)][action[self_action]] + self.alpha*(reward_from_events(self, events)+self.gamma*(self.model[state_to_features(self, new_game_state)][action[self_action]])-self.model[state_to_features(self, old_game_state)][action[self_action]]) #SARSA


        # update score
        ##################
        for (n, s, b, xy) in new_game_state['others']:
            self.score[n] = s
        ##################


    # We want to test in training mode aswell because we have access to the events in this case.
    # But the model should not be updated in then

    #####################
    if config.TRULY_TRAIN:
    #####################
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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(last_game_state, last_action, None, reward_from_events(self, events)))

    #TODO: Where should the reward for a victory be put into?
    score_others = [s for (n, s, b, xy) in last_game_state['others']]
    score_own = last_game_state['self'][1]
    if len(score_others)>0 and score_own>max(score_others):
        events.append(VICTORY)

    #################
    # keep track of score
    for (n, s, b, xy) in last_game_state['others']:
        self.score[n] = s

    scores = np.array(list(self.score.values()))

    # log agents placement at end of the game
    agent_placement = len(scores) - np.searchsorted(np.sort(scores), score_own) + 1
    self.logger.debug(f'Agents placement/score: {agent_placement},{score_own}')
    ##################


    # We want to test in training mode aswell because we have access to the events in this case.
    # But the model should not be updated in then

    ######################
    if config.TRULY_TRAIN:
    ######################
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
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION:-100,
        e.WAITED:-100,
        #e.KILLED_SELF:-500,
        #e.BOMB_DROPPED:-500,
        APPROACH_COIN:1,
        AWAY_FROM_COIN:-2,
        #REPEATED_MOVE:-1,



        }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
