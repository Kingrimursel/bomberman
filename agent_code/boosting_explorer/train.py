import numpy as np
import pickle

from typing import List
from collections import namedtuple, deque

import events as e
import settings as s
from .callbacks import state_to_features, step_to_targets, look_for_targets, danger

# Additional structures
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INDEX = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
    'BOMB': 5,
}
ACTION_ROTATION = {
    1: {0:1, 1:2, 2:3, 3:0, 4:4, 5:5},
    2: {0:1, 1:2, 2:3, 3:0, 4:4, 5:5},
    3: {0:1, 1:2, 2:3, 3:0, 4:4, 5:5}
}
ACTION_SYMMETRY = {
    'HORIZONTAL': {0:2, 1:1, 2:0, 3:3, 4:4, 5:5},
    'VERTICAL': {0:0, 1:3, 2:2, 3:1, 4:4, 5:5},
    'POINT': {0:2, 1:3, 2:0, 3:1, 4:4, 5:5}
}
Transition = namedtuple('Transition',
                        ('old_feature', 'action', 'new_feature', 'reward'))

# Costumized events and rewards for these events.
game_rewards = {
    e.INVALID_ACTION: -100,
    e.WAITED: -25,
    e.COIN_COLLECTED: 5,
    e.COIN_FOUND: 1,
    e.CRATE_DESTROYED: 1,
    e.KILLED_SELF: -100,
    e.BOMB_DROPPED: 0
}
SCALE_COIN    = 1
SCALE_CRATE   = 1
SCALE_DANGER  = 10
SCALE_BOMB    = 4

# Hyperparameter
GAMMA            = 0.8
EPSILON          = 0.35
N_ROUNDS_INITIAL = 2
N_ROUNDS_MAX     = 5
BATCH_SIZE       = 700 * 2**7
N_ESTIMATORS     = 5
N_ESTIMATORS_MAX = 40

# Trainingsparameters
LOW_DENSITY = {
    'CRATE_DENSITY': 0.25,
    'COIN_COUNT': 0,
}
MEDIUM_DENSITY = {
    'CRATE_DENSITY': 0.5,
    'COIN_COUNT': 0,
}
HIGH_DENSITY = {
    'CRATE_DENSITY': 0.75,
    'COIN_COUNT': 0,
}
HIGH_COIN = {
    'CRATE_DENSITY': 0,
    'COIN_COUNT': 50,
}
HIGH_COIN_LOW_DENSITY = {
    'CRATE_DENSITY': 0.25,
    'COIN_COUNT': 50,
}
MIDDLE_COIN_MIDDLE_DENSITY = {
    'CRATE_DENSITY': 0.5,
    'COIN_COUNT': 25,
}
CLASSIC = {
    'CRATE_DENSITY': 0.75,
    'COIN_COUNT': 9,
}


def setup_training(self):
    """
    Initialize self for training purpose.

    :param self: Self object of Class Agent.
    """
    self.epsilon         = EPSILON
    self.gamma           = GAMMA
    self.features        = np.zeros((BATCH_SIZE, 9), dtype=np.double)
    self.rewards         = np.zeros(BATCH_SIZE, dtype=np.float)
    self.track_coins     = set()
    self.rounds_count    = 1
    self.rounds_initial  = N_ROUNDS_INITIAL
    self.rounds_max      = N_ROUNDS_MAX
    self.estimators_more = N_ESTIMATORS
    self.estimators_max  = N_ESTIMATORS_MAX
    self.batch_size      = BATCH_SIZE
    self.batch_count     = 0
    self.strategy        = LOW_DENSITY

    s.SCENARIOS['classic']['CRATE_DENSITY'] = self.strategy['CRATE_DENSITY']
    s.SCENARIOS['classic']['COIN_COUNT']    = self.strategy['COIN_COUNT']

    assert self.rounds_max >= self.rounds_initial, "The number of maximal rounds is bigger than the required inital rounds!"


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: Self object of Class Agent.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Update explored coin tracker.
    if not old_game_state is None:
        for coin in old_game_state['coins']:
            self.track_coins.add(coin)

    # Fix bug in rule_based_agent.
    if self_action is None:
        self_action = 'WAIT'
        events.remove(e.INVALID_ACTION)

    # Update learning strategy.
    if self.rounds_count > 3:
        s.SCENARIOS['classic']['CRATE_DENSITY'] = 0.75

    # Remove events which are results from previous actions.
    while e.CRATE_DESTROYED in events:
        events.remove(e.CRATE_DESTROYED)
        i = self.batch_count%self.batch_size
        self.rewards[i-28:i-21] += game_rewards[e.CRATE_DESTROYED]
        self.logger.debug(f"Removing {e.CRATE_DESTROYED} from events list and adding rewards to {self.rewards[i-28:i-21]}.")
    while e.COIN_FOUND in events:
        events.remove(e.COIN_FOUND)
        i = self.batch_count%self.batch_size
        self.rewards[i-28:i-21] += game_rewards[e.COIN_FOUND]
        self.logger.debug(f"Removing {e.COIN_FOUND} from events list and adding rewards to {self.rewards[i-28:i-21]}.")
    while e.KILLED_SELF in events:
        events.remove(e.KILLED_SELF)
        i = self.batch_count%self.batch_size
        self.rewards[i-21:i]     += game_rewards[e.KILLED_SELF]
        #self.rewards[i-28:i-21] += game_rewards[e.KILLED_SELF]
        self.logger.debug(f"Removing {e.KILLED_SELF} from events list and adding rewards to {self.rewards[i-7:i]}.")


    # Add trainings data.
    if self.rounds_count <= self.rounds_max:
        add_trainings_data(self, old_game_state, self_action, new_game_state, events)

    # Train new ensemble.
    if self.batch_count != 0 and self.batch_count%self.batch_size == 0 and self.rounds_count <= self.rounds_max:
        self.logger.debug(f"Train ensemble in game_events_occurred with {self.batch_count}!")
        train_ensemble(self, old_game_state['round'])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    :param self: Self object of Class Agent.
    :param last_game_state: The state that was passed to the last call of `act`.
    :param last_action: The last action that you took.
    :param events: The events that occurred when going in  `last_game_state.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Pack model.
    with open(self.code_name + ".pt", "wb") as file:
        pickle.dump(self.model, file)


def add_trainings_data(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Add new trainings data to current batch.

    :param self: Self object of Class Agent.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if old_game_state is None:
        self.logger.debug(f"Can not add new trainings data in transition to {new_game_state['step']} with transition action {self_action} because old_game_state is None.")
        return

    self.logger.info(f"Add new trainings data in transition from step {old_game_state['step']} to {new_game_state['step']} with transition action {self_action}.")

    # Determine training features.
    features = get_equivalent_features(state_to_features(self, old_game_state))
    actions  = get_equivalent_actions(ACTION_TO_INDEX[self_action])

    # Determine training targets.
    if self.rounds_count <= self.rounds_initial:
        reward        = reward_from_events(self, events)
        reward_coin   = potential_coin(self, old_game_state, new_game_state)
        reward_danger = potential_danger(self, old_game_state, self_action, new_game_state)
        reward_bomb   = potential_bomb(self, old_game_state, self_action, new_game_state)

        for feature, action in zip(features, actions):
            i = self.batch_count%self.batch_size
            self.features[i] = np.append(feature, action)
            self.rewards[i]  = reward + reward_coin + reward_danger + reward_bomb

            self.batch_count += 1

    else:
        reward        = reward_from_events(self, events)
        reward_coin   = potential_coin(self, old_game_state, new_game_state)
        reward_danger = potential_danger(self, old_game_state, self_action, new_game_state)
        reward_bomb   = potential_bomb(self, old_game_state, self_action, new_game_state)

        for feature, action in zip(features, actions):
            i = self.batch_count%self.batch_size
            self.features[i] = np.append(feature, action)

            actions_in = np.arange(5)
            actions_in.resize((5, 1))
            predictions = self.model.predict(np.append(np.repeat(feature[None], 5, axis=0), actions_in, axis=1))
            prediction  = np.max(predictions)

            self.rewards[i] = reward + reward_coin + reward_danger + reward_bomb + self.gamma*prediction

            self.batch_count += 1


def train_ensemble(self, round: int):
    """
    When called new ensemble gets trained.

    :param self: Self object of Class Agent.
    """
    assert self.batch_count%self.batch_size == 0, f"train_ensemble is called even though batch_count is not right: {self.batch_count}/{self.batch_size}"

    # Fit new estimator.
    if self.model.n_estimators < self.estimators_max:
        self.model.n_estimators += self.estimators_more
    self.model.fit(self.features[:-28], self.rewards[:-28])

    # Check how fitting went.
    predictions = self.model.predict(self.features)
    self.logger.info(f"Error of current prediction: {np.linalg.norm(predictions - self.rewards)}")

    # Reset batch for new one.
    f_store = self.features[-28:]
    r_store = self.rewards[-28:]

    self.features      = np.zeros((self.batch_size, 9), dtype=np.double)
    self.features[:28] = f_store

    self.rewards      = np.zeros(self.batch_size, dtype=np.float)
    self.rewards[:28] = r_store

    # Inform about current parameters of GradientBoostingRegressor.
    self.logger.info(f"step: {round}\nself.rounds_count: {self.rounds_count}\nself.rounds_max: {self.rounds_max}\nself.rounds_initial: {self.rounds_initial}\nn_estimators: {self.model.n_estimators}\nn_features_in: {self.model.n_features_in_}\nfeature_importances_: {self.model.feature_importances_}\nself.rounds_initial: {self.rounds_initial}\nself.batch_count: {self.batch_count}\nself.batch_count%self.batch_size: {self.batch_count%self.batch_size}")

    # Increment rounds count.
    self.rounds_count += 1


def reward_from_events(self, events: List[str]) -> int:
    """
    Analyses the occured events and distribute according to this event list rewards.

    :param self: Self object of Class Agent.
    :paran events: List of occured events in transition from last game state to current game state.
    """
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum


def potential_coin(self, old_game_state: dict, new_game_state: dict) -> float:
    """
    Approximates a potential for collecting coins.

    :param self: Self objetct of Class Agent.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param new_game_state: The state after old game state.
    :return: Difference in potential between both states.
    """
    if len(old_game_state['coins']) == 0:
        return .0

    old_arena = old_game_state['field']
    new_arena = new_game_state['field']
    old_coins = old_game_state['coins']
    old_danger = danger(old_game_state)
    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]

    step_to_coin = step_to_targets(np.logical_and(old_arena == 0, np.logical_not(old_danger)), old_position, old_coins)
    next_coin = look_for_targets(np.logical_and(old_arena == 0, np.logical_not(old_danger)), old_position, old_coins)

    if step_to_coin == new_position:
        self.logger.debug(f"Rewarded for heading to next coin at {next_coin}.")
        reward = SCALE_COIN/(1-self.epsilon)
    else:
        self.logger.debug(f"Penalized for not following the path to next coin at {next_coin}.")
        reward = - SCALE_COIN/(1-self.epsilon)

    self.logger.info(f"Reward {reward} for targeting available coins.")
    return reward


def potential_danger(self, old_game_state: dict, self_action: str, new_game_state: dict) -> float:
    """
    Approcimates a potential for escaping danger.

    :param self: Self object of Class Agent.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The last action that one took in transition from old_game_state to new_game_state.
    :param new_game_state: The game state after old_game_state.
    :param events: The events that occurred when going in new game state.
    :return: Difference in potential between both states.
    """
    if old_game_state is None or new_game_state is None:
        self.logger.debug(f"Reward 0 for handling danger because one of the game state is None.")
        return .0

    if self_action == 'BOMB':
        self.logger.debug(f"Agent recently dropped a bomb. Give some time to leave region of danger.")
        return .0

    old_arena = old_game_state['field']
    new_arena = new_game_state['field']
    old_danger = danger(old_game_state)
    new_danger = danger(new_game_state)
    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]

    step_to_escape = step_to_targets(old_arena == 0, old_position, np.argwhere(np.logical_not(old_danger)))
    next_escape = look_for_targets(old_arena == 0, old_position, np.argwhere(np.logical_not(old_danger)))

    reward = .0
    if old_danger[old_position]:
        if step_to_escape == new_position:
            self.logger.debug(f"Rewarded for escaping danger {step_to_escape} at {next_escape}.")
            reward = SCALE_DANGER/(1-self.epsilon)
        else:
            self.logger.debug(f"Penalized for not following the path {step_to_escape} to escape danger at {next_escape}.")
            reward = - SCALE_DANGER/(1-self.epsilon) * 2
    else:
        if new_danger[new_position]:
            self.logger.debug(f"Penalized for entering region of danger.")
            reward = - SCALE_DANGER/(1-self.epsilon) * 2

    self.logger.debug(f"Reward {reward} for handling danger.")
    return reward


def potential_bomb(self, old_game_state: dict, self_action: str, new_game_state: dict) -> float:
    """
    Reward and penalizing for dropping bombs.

    :param self: Self object of Class Agent.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: Action that was taken in transition from old_game_state to new_game_state.
    :param new_game_state: The state after old game state.
    :return: Difference in potential between both states.
    """
    old_position  = old_game_state['self'][3]
    new_position  = new_game_state['self'][3]
    old_x, old_y  = old_position
    old_arena     = old_game_state['field']
    old_danger    = danger(old_game_state)
    old_no_danger = np.logical_not(old_danger)

    step_to_crate = step_to_targets(old_arena == 0, old_position, np.argwhere(np.logical_and(old_arena == 1, old_no_danger)))
    next_crate    = look_for_targets(old_arena == 0, old_position, np.argwhere(np.logical_and(old_arena == 1, old_no_danger)))
    self.logger.debug(f"step_to_crate: {step_to_crate}, next_crate: {next_crate}")

    crates_count = 0
    walls_count  = 0
    danger_count = int(old_danger[old_position])
    for position in [(old_x, old_y + h) for h in [-1, 1]] + [(old_x + h, old_y) for h in [1, -1]]:
        if old_arena[position] == 1:
            crates_count +=1
        if old_arena[position] == -1:
            walls_count += 1
        if old_danger[position]:
            danger_count += 1

    if danger_count != 0:
        if self_action == 'BOMB':
            self.logger.debug("Penalized agent for placing a bomb even if there is danger next to agent.")
            reward = - SCALE_BOMB/(1-self.epsilon) * 4
        else:
            if step_to_crate == new_position:
                self.logger.debug(f"Heading towards new crate at {next_crate} towards {step_to_crate} even if there is danger in neighborhood.")
                reward = SCALE_CRATE/(1-self.epsilon) * 1/2
            else:
                self.logger.debug(f"Penalized for not heading towards new crate at {next_crate} towards {step_to_crate} even if there is danger in neighborhood.")
                reward = -SCALE_CRATE/(1-self.epsilon) * 1/2
    else:
        if step_to_crate == old_position:
            if self_action == 'BOMB':
                self.logger.debug(f"Rewarded reaching tile {old_position} next to target {next_crate} and placing a bomb.")
                reward = SCALE_BOMB/(1-self.epsilon)
            else:
                self.logger.debug(f"Penalized leaving tile {old_position} next to target {next_crate} without dropping a bomb.")
                reward = - SCALE_BOMB/(1-self.epsilon) * 10
        else:
            if self_action == 'BOMB':
                self.logger.debug(f"Penalized for placing a bomb at {old_position} even if target {next_crate} is not reached yet.")
                reward = - SCALE_CRATE/(1-self.epsilon) * 2
            else:
                if step_to_crate == new_position:
                    self.logger.debug(f"Heading towards new crate at {next_crate} towards {step_to_crate}.")
                    reward = SCALE_CRATE/(1-self.epsilon)
                else:
                    self.logger.debug(f"Penalized for not heading towards new crate at {next_crate} towards {step_to_crate}.")
                    reward = - SCALE_CRATE/(1-self.epsilon) * 4

    self.logger.debug(f"Rewarded {reward} for handling bombs and crates.")
    return reward


def get_equivalent_features(feature_normalized: np.array) -> np.array:
    """
    Uses symmetry of playground to determine equivalent features. There is rotation invariance by ratations
    of pi/2 as well as point symmetry, horizontel symmetry and vertical symmetry.

    :param feature: The feature we want to consider.
    :return: Array of equivalent features in the follwing order
        (action, rotation by pi/2, rotation by pi, rotation by 3pi/2, horizontel reflection, vertical reflection, point reflection)
    """
    assert feature_normalized.shape == (8, ), "Feature has not the expected size. Maybe action index was included."

    features = np.zeros((7, 8), np.double)
    norm     = np.array([3, 3, 3, 3, 3, 5, 5, 5])/100
    feature  = feature_normalized*norm

    features[0] = feature

    feature_tiles  = feature[:4]
    f_up_tile      = feature[0]
    f_right_tile   = feature[1]
    f_down_tile    = feature[2]
    f_left_tile    = feature[3]
    f_origin_tile  = feature[4]
    f_nearest_coin = feature[5]
    f_danger       = feature[6]
    f_crate        = feature[7]

    # Ratation invariance.
    for i in range(1, 4):
        features[i] = np.concatenate((np.roll(feature_tiles, i), np.array([f_origin_tile]), np.array([ACTION_ROTATION[i][f_nearest_coin]]), np.array([ACTION_ROTATION[i][f_danger]]), np.array([ACTION_ROTATION[i][f_crate]])))

    # Reflection invariance: Horizontal reflection.
    feature_h   = np.array([f_down_tile, f_right_tile, f_up_tile, f_left_tile, f_origin_tile, ACTION_SYMMETRY['HORIZONTAL'][f_nearest_coin], ACTION_SYMMETRY['HORIZONTAL'][f_danger], ACTION_SYMMETRY['HORIZONTAL'][f_crate]])
    features[4] = feature_h

    # Vertical reflection.
    feature_v   = np.array([f_up_tile, f_left_tile, f_up_tile, f_right_tile, f_origin_tile, ACTION_SYMMETRY['VERTICAL'][f_nearest_coin], ACTION_SYMMETRY['VERTICAL'][f_danger], ACTION_SYMMETRY['VERTICAL'][f_crate]])
    features[5] = feature_v

    # Point reflection.
    feature_p   = np.array([f_down_tile, f_left_tile, f_up_tile, f_right_tile, f_origin_tile, ACTION_SYMMETRY['POINT'][f_nearest_coin], ACTION_SYMMETRY['POINT'][f_danger], ACTION_SYMMETRY['POINT'][f_crate]])
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
    actions = np.array([action, ACTION_ROTATION[1][action], ACTION_ROTATION[2][action], ACTION_ROTATION[3][action], ACTION_SYMMETRY['HORIZONTAL'][action], ACTION_SYMMETRY['VERTICAL'][action], ACTION_SYMMETRY['POINT'][action]])

    return actions
