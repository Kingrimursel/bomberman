import numpy as np
def setup(self):
    pass


def act(self, game_state: dict):
    # arena = game_state['field']
    # explosion_map = game_state['explosion_map']
    # step = game_state['step']
    # bombs = game_state['bombs']
    # bomb_xys = [xy for (xy, t) in bombs]
    # n,s,b,(x, y) = game_state['self']
    # others = [(n, s, b, xy) for (n, s, b, xy) in game_state['others']] #For calculating the number of coins collected yet
    # coins = game_state['coins']
    # bomb_map = np.ones(arena.shape) * 5
    # for (xb, yb), t in bombs:
    #     for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
    #         if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
    #             bomb_map[i, j] = min(bomb_map[i, j], t)   #Can be used as a measure for danger: 5 is nothing, 0 is sure death
    #
    #
    #
    # print('Bomb Map: '+str(bomb_map))
    # print('Explosion Map: '+str(explosion_map))
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']
