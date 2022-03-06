import numpy as np

from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, get_danger_level, create_future_explisions_map

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PASSIVE_IN_DANGER_EVENT = "PASSIVE_IN_DANGER"
ESCAPED_DANGER_EVENT    = "ESCAPED_DANGER"
MOVED_INTO_DANGER_EVENT = "MOVED_INTO_DANGER"


# action to int dit
ACTIONS = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # at the beginning of a round
    if not old_game_state:
        return


    # TODO: dont stay when in danger, and dont be in danger at the end

    # TODO: add reward when agent is next to a coin

    old_position = old_game_state["self"][-1]
    new_position = new_game_state["self"][-1]

    # TODO: don't do there calculations twice. Save them somehow in the state_to_features callback
    old_future_explosions_map = create_future_explisions_map(np.array(old_game_state["bombs"], dtype=object), old_game_state["field"])
    new_future_explosions_map = create_future_explisions_map(np.array(new_game_state["bombs"], dtype=object), new_game_state["field"])


    old_danger_level = get_danger_level(*old_position, old_future_explosions_map)
    new_danger_level = get_danger_level(*new_position, new_future_explosions_map)

    ## own events to hand out rewards

    # if agent has not moved even though he was in danger
    if old_position == new_position and old_danger_level != 0:
            events.append(PASSIVE_IN_DANGER_EVENT)

    # if agent has escaped a dangerous situation
    if old_danger_level > 0 and new_danger_level == 0:
        events.append(ESCAPED_DANGER_EVENT)

    # agent got into danger from a safe position
    if old_danger_level == 0 and new_danger_level > 0:
        events.append(MOVED_INTO_DANGER_EVENT)


    # calculate current transition
    current_transition = Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(self, events))


    # append current transition to transition stack
    self.transitions.append(current_transition)

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)


    old_model = self.model[old_features]
    new_model = self.model[new_features]



    # do temporal difference learning
    old_model[ACTIONS[self_action]] = old_model[ACTIONS[self_action]] + self.alpha * (
            reward_from_events(self, events) +
            self.gamma * np.max(new_model) -
            old_model[ACTIONS[self_action]]
        )


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

    # append current transition to transition stack
    self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))

    # TODO: do Q-learning here aswell

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 4,
        e.KILLED_OPPONENT: 5,
        e.CRATE_DESTROYED: 2.5,
        e.COIN_FOUND: 2.5,
        e.KILLED_SELF: -6,
        e.OPPONENT_ELIMINATED: 5,
        e.INVALID_ACTION: -2,
        e.SURVIVED_ROUND: 4,
        PASSIVE_IN_DANGER_EVENT: -2,
        ESCAPED_DANGER_EVENT: 4,
        MOVED_INTO_DANGER_EVENT: -2.5
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
