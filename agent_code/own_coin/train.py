import numpy as np
import random
import pickle

from typing import List
from collections import namedtuple, deque

import events as e
from .callbacks import state_to_features, look_for_targets

# Additional structures
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
ACTIONS             = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_TO_INDEX    = {'UP': 0, 'RIGHT': 1, 'DOWN': 2,'LEFT': 3 , 'WAIT': 4, 'BOMB': 5}


# Hyperparameters
TRANSITION_HISTORY_SIZE  = 3
ALPHA                    = .4
GAMMA                    = .4
EPSILON                  = .2
RECORD_ENEMY_TRANSITIONS = 1.0

# Events
APPROACH_COIN  = "APPROACH_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
VICTORY        = "VICTORY"


def setup_training(self):
    """
    Initialize self for training purpose.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.alpha   = ALPHA
    self.gamma   = GAMMA
    self.epsilon = EPSILON

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def update_Q():


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state == None:
        return

    # Define old game state roperties.
    old_arena      = old_game_state['field']
    _, _, _,(old_x, old_y) = old_game_state['self']
    old_coins      = old_game_state['coins']
    old_free_space = old_arena == 0

    # Define new game state properties.
    new_arena      = new_game_state['field']
    _, _, _,(new_x, new_y) = new_game_state['self']
    new_coins      = new_game_state['coins']
    new_free_space = new_arena == 0

    # Own events to hand out rewards.
    if len(new_coins)>0 and look_for_targets(old_free_space, (old_x, old_y), old_coins, dir=True)[1] == (new_x, new_y):
        events.append(APPROACH_COIN)
    if len(new_coins)>0 and look_for_targets(old_free_space, (old_x, old_y), old_coins, dir=True)[1] != (new_x, new_y):
        events.append(AWAY_FROM_COIN)

    # Determine rewards and feature indices.
    feature = state_to_features(self, old_game_state)
    action  = ACTIONS_TO_INDEX[self_action]
    rewards = reward_from_events(self, events)

    #N-Step Q Learning
    #current_reward = reward_from_events(self, events)
    #
    #old_states   = np.array([old_state for (old_state,action,new_state,reward) in self.transitions])
    #actions      = np.array([ACTIONS[a] for (old_state,a,new_state,reward) in self.transitions])
    #new_states   = np.array([new_state for (old_state,action,new_state,reward) in self.transitions])
    #prev_rewards = np.array([reward for (old_state,action,new_state,reward) in self.transitions])
    #
    #old_states   = np.append(old_states, old_game_state)
    #actions      = np.append(actions, ACTIONS_TO_INDEX[self_action])
    #new_states   = np.append(new_states, new_game_state)
    #prev_rewards = np.append(prev_rewards, current_reward)
    #
    #gamma_exp = np.array([self.gamma**k for k in range(0,len(actions))])
    #
    #self.model[state_to_features(self, old_states[0])][[0]] = self.model[state_to_features(self, old_states[0])][actions[0]] + self.alpha*(np.sum(gamma_exp*prev_rewards +(self.gamma**TRANSITION_HISTORY_SIZE)*np.max(self.model[state_to_features(self, new_game_state)]))-self.model[state_to_features(self, old_states[0])][actions[0]])
    #self.logger.info(f"Model updated")

    # Q-Learning
    self.model[feature][action] = self.model[feature][action] + self.alpha*(rewards + self.gamma*(np.max(self.model[feature])) - self.model[feature][action])

    # SARSA
    #self.model[state_to_features(self, old_game_state)][action[self_action]] = self.model[state_to_features(self, old_game_state)][action[self_action]] + self.alpha*(reward_from_events(self, events)+self.gamma*(self.model[state_to_features(self, new_game_state)][action[self_action]])-self.model[state_to_features(self, old_game_state)][action[self_action]]) #SARSA

    with open(self.code_name + ".pt", "wb") as file:
        pickle.dump(self.model, file)

    self.transitions.append(Transition(old_game_state, self_action, new_game_state, reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: Current game state.
    :param last_action: Action that was took in last step.
    :param events: Events that occured in transition to last_game_stat.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Add last transition.
    self.transitions.append(Transition(last_game_state, last_action, None, reward_from_events(self, events)))

    #TODO: Where should the reward for a victory be put into?
    score_others = [s for (n, s, b, xy) in last_game_state['others']]
    score_own    = last_game_state['self'][1]
    if len(score_others) > 0 and score_own > max(score_others):
        events.append(VICTORY)

    # Store updated model
    with open(self.code_name + ".pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]):
    """
    Here you can modify the rewards your agent get so as to en/discourage certain behavior.

    :param self: The same object that was passed to all other functions.
    :param events: List of events that need to consider to reward actions.
    """
    #TODO: Create good rewards/penalties with good values. Avoid repeating moves somehow.
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION:-100,
        e.WAITED:-100,
        #e.KILLED_SELF:-500,
        #e.BOMB_DROPPED:-500,
        APPROACH_COIN:1,
        AWAY_FROM_COIN:-2,
        }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum
