import matplotlib
import numpy as np
import pandas as pd

from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStates = namedtuple("States", ["episode_lengths", "episode_rewards"])

def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot
    :param V: matrix of state value function
    :param title: the plot title name
    :return: non
    """

    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)

    # The purpose of meshgrid is to create a rectangular grid out of an array of x values and an array of y values.
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x,y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[_[0], _[1], True], 2, np.dstack([X,Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        #ax.set_zlable('Value')

        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usabel Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

