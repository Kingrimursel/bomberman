import numpy as np
import pickle

from typing import List
from collections import namedtuple, deque

import events as e
from .callbacks import state_to_features, look_for_targets

# Additional structures
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
ACTION_TO_INDEX = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
}
ACTION_SYMMETRY = {
    'HORIZONTAL': {0:2, 1:1, 2:0, 3:3, 4:4, 5:5},
    'VERTICAL': {0:0, 1:3, 2:2, 3:1, 4:4, 5:5},
    'POINT': {0:2, 1:3, 2:0, 3:1, 4:4, 5:5}
}
Transition = namedtuple('Transition',
                        ('old_feature', 'action', 'new_feature', 'reward'))

# Hyperparameter
GAMMA          = 0.8
EPSILON        = 0.2
INITIAL_ROUNDS = 1
BATCH_SIZE     = 14000


def setup_training(self):
    """
    Initialize self for training purpose.

    :param self: Self object of Class Agend.
    """
    self.epsilon     = EPSILON
    self.gamma       = GAMMA
    self.features    = []
    self.rewards     = []
    self.batch_size  = BATCH_SIZE
    self.batch_count = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: Self object of Class Agend.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    add_trainings_data(self, old_game_state, self_action, new_game_state, events)

    if self.batch_count != 0 and self.batch_count%self.batch_size == 0:
        train_ensemble(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    :param self: Self object of Class Agend.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    if self.batch_count != 0 and self.batch_count%self.batch_size == 0:
        train_ensemble(self)

    with open(self.code_name + ".pt", "wb") as file:
        pickle.dump(self.model, file)


def add_trainings_data(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Add new trainings data to current batch.

    :param self: Self object of Class Agend.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if old_game_state is None:
        return

    # Determine training features.
    features = get_equivalent_features(state_to_features(self, old_game_state))
    actions  = get_equivalent_actions(ACTION_TO_INDEX[self_action])

    for feature, action in zip(features, actions):
        self.features.append(np.append(feature, action))

    # Determine training targets.
    if self.initial_rounds > 0:
        old_free_space = old_game_state['field'] == 0
        old_start      = old_game_state['self'][3]
        old_targets    = old_game_state['coins']
        target         = look_for_targets(old_free_space, old_start, old_targets, self.logger)

        reward    = reward_from_events(self, events)
        potential = potential_coin(self, old_game_state, target, new_game_state)

        for feature in features:
            self.logger.debug(f"Adding {reward} + {potential} to reward list.")
            self.rewards.append(reward + potential)

    else:
        old_free_space = old_game_state['field'] == 0
        old_start      = old_game_state['self'][3]
        old_targets    = old_game_state['coins']
        target         = look_for_targets(old_free_space, old_start, old_targets, self.logger)

        reward    = reward_from_events(self, events)
        potential = potential_coin(self, old_game_state, target, new_game_state)

        for feature in features:
            actions = np.arange(5)
            actions.resize((5, 1))

            predictions = self.model.predict(np.append(np.repeat(feature[None], 5, axis=0), actions, axis=1))
            prediction  = np.max(predictions)

            self.rewards.append(reward + potential + self.gamma*prediction)

    self.batch_count += 7


def train_ensemble(self):
    """
    When called new ensemble gets trained.

    :param self: Self object of Class Agend.
    """
    self.logger.debug(f"Batch size is reached: {self.batch_count}")
    assert self.batch_count%self.batch_size == 0, f"train_ensemble is called even though batch_count is not right: {self.batch_count}/{self.batch_size}"

    # Increment initial rounds count.
    self.logger.debug(f"Initial round: {self.initial_rounds}")
    self.initial_rounds -= 1

    # Fit new estimator.
    self.logger.debug(f"Updating model. Current number of estimators: {self.model.n_estimators}")
    self.logger.debug(self.features)
    self.logger.debug(self.rewards)
    self.model.fit(self.features, self.rewards)
    self.model.n_estimators += 20

    # Check how fitting went.
    predictions = self.model.predict(self.features)
    self.logger.debug(f"Error of current prediction: {np.linalg.norm(predictions - self.rewards)}")

    # Reset batch for new one.
    self.features    = []
    self.rewards     = []
    self.batch_count = 0

    # Inform about current parameters of GradientBoostingRegressor.
    self.logger.info(f"n_estimators: {self.model.n_estimators}/nn_features_in: {self.model.n_features_in_}")


def reward_from_events(self, events: List[str]) -> int:
    """
    Analyses the occured events and distribute according to this event list rewards.

    :param self: Self object of Class Agend.
    :paran events: List of occured events in transition from last game state to current game state.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.INVALID_ACTION: -100,
        e.WAITED: -20,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def potential_coin(self, old_game_state: dict, target: tuple, new_game_state:dict) -> float:
    """
    Approximate a potetnial for coin collecting.

    :param self: Self objetct of Class Agend.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param target: Tuple of coordinates of the target.
    :param new_game_state: The state after old game state.
    """
    old_position = np.array(old_game_state['self'][3])
    new_position = np.array(new_game_state['self'][3])

    Psi_new_state = -np.sum(np.abs(new_position - np.array(target)))/(1-self.epsilon)
    Psi_old_state = -np.sum(np.abs(old_position - np.array(target)))/(1-self.epsilon)

    return self.gamma*Psi_new_state - Psi_old_state


def get_equivalent_features(feature_normalized: np.array) -> np.array:
    """
    Uses symmetry of playground to determine equivalent features. There is rotation invariance by ratations
    of pi/2 as well as point symmetry, horizontel symmetry and vertical symmetry.

    :param feature: The feature we want to consider.
    :return: Array of equivalent features in the follwing order
        (action, rotation by pi/2, rotation by pi, rotation by 3pi/2, horizontel reflection, vertical reflection, point reflection)
    """
    assert feature_normalized.shape == (5, ), "Feature has not the expected size. Maybe action index was included."

    features = np.empty((7, 5))
    norm     = np.array([2, 2, 2, 2, 4])/100
    feature  = feature_normalized*norm

    features[0] = feature

    feature_tiles        = feature[:4]
    f_up_tile            = feature[0]
    f_right_tile         = feature[1]
    f_down_tile          = feature[2]
    f_left_tile          = feature[3]
    feature_nearest_coin = feature[4]

    # Ratation invariance.
    for i in range(1, 4):
        features[i] = np.concatenate((np.roll(feature_tiles, i), np.array([(feature_nearest_coin+i)%4])))

    # Reflection invariance: Horizontal reflection.
    feature_h   = np.array([f_down_tile, f_right_tile, f_up_tile, f_left_tile, ACTION_SYMMETRY['HORIZONTAL'][feature_nearest_coin]])
    features[4] = feature_h

    # Vertical reflection.
    feature_v   = np.array([f_up_tile, f_left_tile, f_up_tile, f_right_tile, ACTION_SYMMETRY['VERTICAL'][feature_nearest_coin]])
    features[5] = feature_v

    # Point reflection.
    feature_p   = np.array([f_down_tile, f_left_tile, f_up_tile, f_right_tile, ACTION_SYMMETRY['POINT'][feature_nearest_coin]])
    features[6] = feature_p

    return features/norm


def get_equivalent_actions(action: int) -> np.array:
    """
    Uses symmetry of playground to determine equivalent features. There is rotation invariance by ratations
    of pi/2 as well as point symmetry, horizontel symmetry and vertical symmetry.

    :param feature: The feature we want to consider.
    :return: Array of equivalent actions in the order
        (action, rotation by pi/2, rotation by pi, rotation by 3pi/2, horizontel reflection, vertical reflection, point reflection)
    """
    actions = np.array([action%4, (action+1)%4, (action+2)%4, (action+3)%4, ACTION_SYMMETRY['HORIZONTAL'][action], ACTION_SYMMETRY['VERTICAL'][action], ACTION_SYMMETRY['POINT'][action]])

    return actions
