import pickle

from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e

from .callbacks import state_to_features, create_training_batch, get_Q_values, look_for_targets

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS_TO_INDEX    = {'UP': 0, 'RIGHT': 1, 'DOWN': 2,'LEFT': 3 , 'WAIT': 4, 'BOMB': 5}

# Hyper parameters -- DO modify
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
APPROACH_COIN  = "APPROACH_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
VICTORY        = "VICTORY"

# TODO: Funktion update_q_table für diese Anwendung anpassen
# TODO: Symmetrien berücksichtigen

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=self.BATCH_SIZE)


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

    # Determine rewards and feature indices.
    feature = state_to_features(self, old_game_state)
    action  = ACTIONS_TO_INDEX[self_action]
    rewards = reward_from_events(self, events)

    # Add current transition.
    self.transitions.append(Transition(old_game_state, self_action, new_game_state, rewards))

    ## # state_to_features is defined in callbacks.py
    ## self.transitions.append(Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(self, events)))


    # do new training step after "BATCH_SIZE" many actions
    if new_game_state["step"] % self.BATCH_SIZE == 0:
        gradient_update(self)



def gradient_update(self):
    """
    Function that does a gradient update on our model
    """


    # create training batches from transition stack
    feature_batches, reward_batches = create_training_batch(self)

    for action in self.ACTIONS:
        feature_batch = feature_batches[action]
        reward_batch  = reward_batches[action]


        for i, features in enumerate(feature_batch):
            reward = reward_batch[i]

            # calculate beta update
            y = reward + self.gamma*np.max(get_Q_values(self, features))
            beta_update = (y - features @ self.model[action])*features

            # actually update beta
            self.model[action] += self.alpha/len(feature_batch)*beta_update


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

        

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
    #self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))

    # Add last transition.
    self.transitions.append(Transition(last_game_state, last_action, None, reward_from_events(self, events)))

    # do a last gradient update
    gradient_update(self)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
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
