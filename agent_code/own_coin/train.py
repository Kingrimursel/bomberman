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


def update_Q(self, nstep=True, SARSA=False, Qlearning=False):
    """
    Performs update on Q depending on which method was chosen. By default N-step SARSA Learning is chosen.

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

        feature_0 = state_to_features(self, transition_0[0])
        feature_n = state_to_features(self, transition_n[0])
        action_0  = ACTIONS_TO_INDEX[transition_0[1]]
        action_n  = ACTIONS_TO_INDEX[transition_n[1]]

        self.logger.debug(f'before: {self.model[feature_0][action_0]}')

        prev_rewards = np.array([r for (_, _, _, r) in self.transitions])
        gamma_pow    = np.power(np.ones(n) * self.gamma, np.arange(n))

        self.model[feature_0][action_0] = self.model[feature_0][action_0] + self.alpha * np.sum(gamma_pow*prev_rewards + self.gamma**n * np.max(self.model[feature_n]) - self.model[feature_0][action_0])

        self.logger.debug(f'after: {self.model[feature_0][action_0]}')

    elif SARSA:
        transition  = self.transitions.popleft()

        old_feature = state_to_features(self, transition[0])
        new_feature = state_to_features(self, transition[2])
        action      = ACTIONS_TO_INDEX[transition[1]]
        rewards     = transition[3]

        if new_feature is None or old_feature is None:
            return

        self.model[old_feature][action] = self.model[old_feature][action] + self.alpha*(rewards + self.gamma*(self.model[new_feature][action]) - self.model[old_feature][action])

    elif Qlearning:
        transition = self.transitions.popleft()
        feature    = state_to_features(self, transition[0])
        action     = ACTIONS_TO_INDEX[transition[1]]
        rewards    = transition[3]

        self.model[feature][action] = self.model[feature][action] + self.alpha*(rewards + self.gamma*np.max(self.model[feature]) - self.model[feature][action])


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

    # Determine rewards and feature indices.
    feature = state_to_features(self, old_game_state)
    action  = ACTIONS_TO_INDEX[self_action]
    rewards = reward_from_events(self, events)

    # Add current transition.
    self.transitions.append(Transition(old_game_state, self_action, new_game_state, rewards))

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

    # Add last transition.
    self.transitions.append(Transition(last_game_state, last_action, None, reward_from_events(self, events)))

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
