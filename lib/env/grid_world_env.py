# -*- coding: utf-8 -*-

import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    def __init__(self, shape=[4,4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('Shape argument must be a list/tuple of length 2')

        self.shape = shape
        # Return the product of array elements over a given axis.from
        # default environment shape is 4 * 4, thus the number of state is 16
        self.nS = np.prod(shape)
        self.nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}

        grid = np.arange(self.nS).reshape(shape)

        # nditer is Efficient multi-dimensional iterator object to iterate over arrays.
        # “multi_index” causes a multi-index, or a tuple of indices with one per iteration dimension, to be tracked.

        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = { a: [] for a in range(self.nA)}

            # To judge if it is a terminal state
            is_done = lambda s: s == 0 or s == (self.nS - 1)

            reward = 0.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                # State transition after a specific action
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1

                #prob next state, reward, done
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()


        # Initial state distribution is uniform
        self.isd = np.ones(self.nS) / self.nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        # call base class construct function
        super(GridworldEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # output current state
            if self.s == s:
                output = "x "
            # Terminal state
            elif s == 0 or s == self.nS - 1:
                output = "T "
            else:
                output = "o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()


# test envs

# grid_env = GridworldEnv()
# print(grid_env.nS)
# grid_env._render()




