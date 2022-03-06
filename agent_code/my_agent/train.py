import numpy as np

from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

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

    if not old_game_state:  # TODO why can this be true?
        print("yees")
        return

    # TODO: add reward when agent is next to a coin

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)


    # calculate current transition
    current_transition = Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(self, events))


    # append current transition to transition stack
    self.transitions.append(current_transition)

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    #print(new_features)
    #print(old_game_state)
    #print(self.model.shape)
    #print(old_features)
    #print(self.model[old_features].shape)

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
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 2,
        e.KILLED_SELF: -4,
        e.OPPONENT_ELIMINATED: 3,
        e.SURVIVED_ROUND: 4,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
