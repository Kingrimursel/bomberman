from collections import namedtuple, deque


import numpy as np
import random
import pickle
from typing import List
import events as e
from .callbacks import state_to_features, look_for_targets, destruction, potential_bomb, bombmap_and_freemap

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only ... last transitions
#RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
APPROACH_COIN = "APPROACH_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
VICTORY = "VICTORY"
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
STUPID_BOMB = "STUPID_BOMB"
STUPID_MOVE ="STUPID_MOVE"
WRONG_DIRECTION="WRONG_DIRECTION"
BOMB_CHANCE_MISSED ="BOMB_CHANCE_MISSED"



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
    if(self_action==None): #Must do this in trainng here, because wait is sometimes returned as None, which crashes the training
        self_action= 'WAIT'
        events.remove(e.INVALID_ACTION)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    #Define hyperparameters for training (adapt for each case)
    alpha=0.1
    gamma=0.8
    self.logger.info(f"Action: {self_action}")
    #if(old_game_state==None): self.logger.info(f"Old Game State is None")

    #old_game_state is none in the first instance (discussed also in discord) because initial state is defined after game_events_occured is called, for whatever reason.
    if(old_game_state != None and new_game_state != None):
        #Define old game state roperties TODO: Do not yet need all of them but probably for finetuning the rewardshaping
        old_arena = old_game_state['field']
        old_step = old_game_state['step']
        old_n,old_s,old_b,(old_x, old_y) = old_game_state['self']
        old_bombs = old_game_state['bombs']
        old_coins = old_game_state['coins']
        cols = range(1, old_arena.shape[0] - 1)
        rows = range(1, old_arena.shape[0] - 1)
        walls = [(x, y) for x in cols for y in rows if (old_arena[x, y] == -1)]
        old_free_space, old_bomb_map = bombmap_and_freemap(old_arena, old_bombs)

        #Define new game state properties TODO: Do not yet need all of them but probably for finetuning the rewardshaping
        new_arena = new_game_state['field']
        new_step = new_game_state['step']
        new_n,new_s,new_b,(new_x, new_y) = new_game_state['self']
        new_coins = new_game_state['coins']
        new_free_space = new_arena == 0 #For the function

        old_features = state_to_features(self, old_game_state)   #Features are sometimes ambigously and depend on BFS, that is why in rare cases the rewards to not match the action, but this is smoothed over time
        self.logger.info(f"Features: {old_features}")

        # Own events to hand out rewards
        #Coins and waits:
        if(len(old_coins)>0 ):
            coin_condition =  look_for_targets(old_free_space, (old_x, old_y), old_coins,dir=True)[1]==(new_x, new_y) #have to define because of BFS this sometimes just switches
            coin_reachable = look_for_targets(old_free_space, (old_x, old_y), old_coins,dir=True)[2]
        if len(old_coins)>0 and coin_condition and coin_reachable:
            events.append(APPROACH_COIN)
        if len(old_coins)>0 and not coin_condition and old_features[5]==0 and coin_reachable and self_action != 'BOMB':
            events.append(AWAY_FROM_COIN)

        if (old_features[0]==old_features[1]==old_features[2]==old_features[3]==1 and self_action=='WAIT'):
            events.append(CORRECT_WAIT)
        if ((old_features[0]==0 or old_features[0]==3 or old_features[1]==0 or old_features[1]==3 or old_features[2]==0 or old_features[2]==3 or old_features[3]==0 or old_features[3]==3) and self_action=='WAIT'):
            events.append(WRONG_WAIT)
        #MOVES:
        if (old_features[0]==3 and self_action=='LEFT'):
            events.append(CORRECT_DIRECTION)
        if (old_features[1]==3 and self_action=='RIGHT'):
            events.append(CORRECT_DIRECTION)
        if (old_features[2]==3 and self_action=='UP'):
            events.append(CORRECT_DIRECTION)
        if (old_features[3]==3 and self_action=='DOWN'):
            events.append(CORRECT_DIRECTION)


        #Only penalize if no chance taken
        if (old_features[0]==3 and self_action!='LEFT' and old_features[1]!=3 and old_features[2]!=3 and old_features[3]!=3):
            events.append(WRONG_DIRECTION)
        if (old_features[1]==3 and self_action!='RIGHT'and old_features[0]!=3 and old_features[2]!=3 and old_features[3]!=3):
            events.append(WRONG_DIRECTION)
        if (old_features[2]==3 and self_action!='UP' and old_features[0]!=3 and old_features[1]!=3 and old_features[3]!=3):
            events.append(WRONG_DIRECTION)
        if (old_features[3]==3 and self_action!='DOWN' and old_features[0]!=3 and old_features[1]!=3 and old_features[2]!=3):
            events.append(WRONG_DIRECTION)


        #BOMBS:
        if (old_features[4]==1 and self_action =='BOMB' ):
            events.append(UNNECCESSARY_BOMB)
        if (old_features[4]==2 and self_action =='BOMB' ):
            events.append(GOOD_BOMB)
        if (old_features[4]==3 and self_action =='BOMB' ):
            events.append(BEST_BOMB)
        if (old_features[4]==0 and self_action =='BOMB'):
            events.append(STUPID_BOMB)

        if ((old_features[0]==3 or old_features[1]==3 or old_features[2]==3 or old_features[3]==3) and self_action =='BOMB' and old_bomb_map[old_x,old_y]==5):
            events.append(BETTER_BOMB_POSSIBLE)
        if ((old_features[0]!=3 and old_features[1]!=3 and old_features[2]!=3 and old_features[3]!=3) and (old_features[4]==2 or old_features[4]==3) and self_action!='BOMB'):
            events.append(BOMB_CHANCE_MISSED)



        if (old_features[4]==4 and self_action =='WAIT'):
            events.append(STUPID_WAIT)

        #BOMB DROP was 4 moves before explosion, reward that if survived


        #N-Step Q Learning:
        # if(len(self.transitions)>=TRANSITION_HISTORY_SIZE):
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
        #     gamma_exp = np.array([gamma**k for k in range(0,len(actions))])
        #
        #
        #
        #     self.model[state_to_features(self, old_states[0])][actions[0]] = self.model[state_to_features(self, old_states[0])][actions[0]] + alpha*(np.sum(gamma_exp*prev_rewards +(gamma**TRANSITION_HISTORY_SIZE)*np.max(self.model[state_to_features(self, new_game_state)]))-self.model[state_to_features(self, old_states[0])][actions[0]])
        #     self.logger.info(f"Model updated")

        #Update Q-function(model) here
        #Q-Learning:
        if (e.BOMB_EXPLODED in events): #If bomb exploded, rewards for the ste where bomb was placed
            bomb_drop_state = self.transitions[0][0] #Exactly the first entry in transitions, since four transitions are saved
            if(bomb_drop_state!=None ):
                drop_state_features = state_to_features(self, bomb_drop_state)
                self.model[drop_state_features][5] = self.model[drop_state_features][5] + alpha*(reward_from_events(self, [ev for ev in events if (ev==e.CRATE_DESTROYED or ev==e.OPPONENT_ELIMINATED  or ev==e.COIN_FOUND)])+gamma*(np.max(self.model[state_to_features(self, self.transitions[1][0])]))-self.model[drop_state_features][5]) #Only rewarding destruction, not the fat that a bomb exploded

        self.model[old_features][action[self_action]] = self.model[old_features][action[self_action]] + alpha*(reward_from_events(self, [ev for ev in events if (ev!=e.CRATE_DESTROYED and ev!=e.COIN_FOUND and ev!= e.OPPONENT_ELIMINATED) ])+gamma*(np.max(self.model[state_to_features(self, new_game_state)]))-self.model[old_features][action[self_action]]) #Q-Learning
        #SARSA:
        #self.model[state_to_features(self, old_game_state)][action[self_action]] = self.model[state_to_features(self, old_game_state)][action[self_action]] + alpha*(reward_from_events(self, events)+gamma*(self.model[state_to_features(self, new_game_state)][action[self_action]])-self.model[state_to_features(self, old_game_state)][action[self_action]]) #SARSA



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
    #self.transitions.append(Transition(last_game_state, last_action, None, reward_from_events(self, events)))

    alpha=0.1
    gamma=0.8

    #VICTORY REWARD #FIXME: Where should this be used? And: Does not work this way, dead opponents are not in 'others' anymore
    score_others = [s for (n, s, b, xy) in last_game_state['others']]
    score_own = last_game_state['self'][1]
    if len(score_others)>0 and score_own>max(score_others):
        events.append(VICTORY)

    old_features = state_to_features(self, last_game_state)
    #Update Q-function(model) here
    #Q-Learning:


    #Bomb drop has been 4 moves earlier, also reward that one.
    if (e.BOMB_EXPLODED in events and e.KILLED_SELF not in events):
        bomb_drop_state = self.transitions[0][0] #Exactly the first entry in transitions, since four transitions are saved
        if(bomb_drop_state!=None ):
            drop_state_features = state_to_features(self, bomb_drop_state)
            self.model[drop_state_features][5] = self.model[drop_state_features][5] + alpha*(reward_from_events(self, [ev for ev in events if (ev==e.CRATE_DESTROYED or ev==e.OPPONENT_ELIMINATED or ev==e.COIN_FOUND)])+gamma*(np.max(self.model[state_to_features(self, self.transitions[1][0])]))-self.model[drop_state_features][5])


    #Q-Learning
    self.model[old_features][action[last_action]] = self.model[old_features][action[last_action]] + alpha*(reward_from_events(self, [ev for ev in events if (ev!=e.CRATE_DESTROYED and ev!= e.OPPONENT_ELIMINATED and ev!=e.COIN_FOUND) ]))#Only this part because new state does not exist. +gamma*(np.max(self.model[state_to_features(self, new_game_state)]))-self.model[old_features][action[self_action]]) #Q-Learning
    #SARSA:
    #self.model[state_to_features(self, old_game_state)][action[self_action]] = self.model[state_to_features(self, old_game_state)][action[self_action]] + alpha*(reward_from_events(self, events)+gamma*(self.model[state_to_features(self, new_game_state)][action[self_action]])-self.model[state_to_features(self, old_game_state)][action[self_action]]) #SARSA

    # Store updated model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    #TODO: Create good rewards/penalties with good values


    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION:-100,
        e.KILLED_SELF:-100,
        e.CRATE_DESTROYED: 1,
        e.OPPONENT_ELIMINATED: 5,
        APPROACH_COIN:1,
        e.COIN_FOUND: 1,
        AWAY_FROM_COIN:-2,
        CORRECT_WAIT:1,
        WRONG_WAIT:-1,
        CORRECT_DIRECTION:1,
        WRONG_DIRECTION:-2,
        BOMB_CHANCE_MISSED:-2,
        GOOD_BOMB:1,
        UNNECCESSARY_BOMB:-2,
        BETTER_BOMB_POSSIBLE:-2,
        BEST_BOMB:6,
        STUPID_BOMB:-100,
        STUPID_WAIT:-100,
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
