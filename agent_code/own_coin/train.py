import sys
import os
import numpy as np
import random
import pickle

from typing import List
from collections import namedtuple, deque

import events as e
from .callbacks import state_to_features, look_for_targets

sys.path.append(os.path.abspath(".."))

from agent_code.own_coin import config

# Additiona Structures
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
ACTIONS             = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_TO_INDEX    = {'UP': 0, 'RIGHT': 1, 'DOWN': 2,'LEFT': 3 , 'WAIT': 4, 'BOMB': 5}
ACTION_SYMMETRY     = {
    'HORIZONTAL': {0:2, 1:1, 2:0, 3:3, 4:4, 5:5},
    'VERTICAL': {0:0, 1:3, 2:2, 3:1, 4:4, 5:5},
    'POINT': {0:2, 1:3, 2:0, 3:1, 4:4, 5:5}
}

# Hyperparameters
TRANSITION_HISTORY_SIZE  = 3
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
    self.alpha   = config.ALPHA
    self.gamma   = config.GAMMA
    self.epsilon = config.EPSILON

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def get_equivalent_features(feature: tuple):
    """
    Using symmetries of the playground to get equivalent feature of the passed feature.

    :param feature: A tuple of the indices in feature space/self.model except the last axis
        which indicates the action to take in this state. Remember that the feature indeices are
        build up like this: (left, right, above, below, game mode).
    :return: Three equivalent features in the follwoing order:
        (horizontal reflection, vertical reflectino, point reflection)
    """
    f_left    = feature[0]
    f_right   = feature[1]
    f_below   = feature[2]
    f_above   = feature[3]
    f_current = feature[4]
    f_phase   = feature[5]

    # horizontal reflection
    feature_h = tuple([f_left, f_right, f_above, f_below, f_current, f_phase])
    # vertiacl reflection
    feature_v = tuple([f_right, f_left, f_below, f_above, f_current, f_phase])
    # point reflection
    feature_p = tuple([f_right, f_left, f_above, f_below, f_current, f_phase])

    return [feature, feature_h, feature_v, feature_p]


def get_equivalent_actions(action: int):
    """
    Using symmetries of the playground to get equivalent feature of the passed feature.

    :param action: The index of the last axis in feature space/self.model.
    :return: Three equivalent features in the follwoing order:
        (horizontal reflection, vertical reflectino, point reflection)
    """
    # horizontal reflection
    action_h  = ACTION_SYMMETRY['HORIZONTAL'][action]
    # vertiacl reflection
    action_v  = ACTION_SYMMETRY['VERTICAL'][action]
    # point reflection
    action_p  = ACTION_SYMMETRY['POINT'][action]

    return np.array([action, action_h, action_v, action_p])


def update_Q(self, nstep=True, SARSA=False, Qlearning=False):
    """
    Performs update on Q depending on which method was chosen. By default N-step Q-learning is chosen.

    :param self: This object is passed to all callbacks.
    :param nstep/SARSA/Qlearning: Boolean which indicates which update method should be chosen.
    """
    if np.sum([nstep, Qlearning, SARSA]) != 1:
        raise TypeError(f"Choose only and at least one learning method. {np.sum([nstep, Qlearning, SARSA])} were given.")

    if nstep:
        transition_0 = self.transitions.popleft()
        n            = len(self.transitions)

        if transition_0[0]['step']<TRANSITION_HISTORY_SIZE or n<1:
            return

        transition_n = self.transitions.pop()
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
        transition  = self.transitions.popleft()

        if transition[0] is None or transition[2] is None:
            return

        old_feature = get_equivalent_features(state_to_features(self, transition[0]))
        new_feature = get_equivalent_features(state_to_features(self, transition[2]))
        action      = get_equivalent_actions(ACTIONS_TO_INDEX[transition[1]])
        rewards     = transition[3]

        for i in range(4):
            self.model[old_feature[i]][action[i]] = self.model[old_feature[i]][action[i]] + self.alpha*(rewards + self.gamma*(self.model[new_feature[i]][action[i]]) - self.model[old_feature[i]][action[i]])

    elif Qlearning:
        transition = self.transitions.popleft()
        feature    = get_equivalent_features(state_to_features(self, transition[0]))
        action     = get_equivalent_actions(ACTIONS_TO_INDEX[transition[1]])
        rewards    = transition[3]

        self.logger.debug(f'feature: {feature}')
        self.logger.debug(f'transition: {state_to_features(self, transition[0])}')
        self.logger.debug(f'action: {action}')
        self.logger.debug(f'transition action: {ACTIONS_TO_INDEX[transition[1]]}')

        for i in range(4):
            self.model[feature[i]][action[i]] = self.model[feature[i]][action[i]] + self.alpha*(rewards + self.gamma*np.max(self.model[feature[i]]) - self.model[feature[i]][action[i]])


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
    step           = old_game_state['step']

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

    # update score
    for (n, s, b, xy) in new_game_state['others']:
        self.score[n] = s

    if config.DETERMINISTIC and not self_action:
        self_action = 'WAIT'

    # Determine rewards and feature indices.
    feature = state_to_features(self, old_game_state)
    action  = ACTIONS_TO_INDEX[self_action]
    rewards = reward_from_events(self, events)

    # Add current transition.
    self.transitions.append(Transition(old_game_state, self_action, new_game_state, rewards))

    if config.TRULY_TRAIN:
        # Update Q matrix.
        update_Q(self, nstep=True, SARSA=False, Qlearning=False)

        with open(self.code_name + ".pt", "wb") as file:
            pickle.dump(self.model, file)


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

    # keep track of score
    for (n, s, b, xy) in last_game_state['others']:
        self.score[n] = s

    scores = np.array(list(self.score.values()))

    # log agents placement at end of the game
    agent_placement = len(scores) - np.searchsorted(np.sort(scores), score_own) + 1
    self.logger.debug(f'Agents placement/score: {agent_placement},{score_own}')

    # Add last transition.
    self.transitions.append(Transition(last_game_state, last_action, None, reward_from_events(self, events)))

    if config.TRULY_TRAIN:
        # Make last updates on Q.
        while len(self.transitions)>1:
            update_Q(self, nstep=True, SARSA=False, Qlearning=False)

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
        e.INVALID_ACTION: -100,
        e.WAITED: -100,
        APPROACH_COIN: 1,
        AWAY_FROM_COIN: -2,
        }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum
