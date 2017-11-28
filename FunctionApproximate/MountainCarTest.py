import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.envs.make("MountainCar-v0")

env.reset()
plt.figure()
plt.imshow(env.render(mode='rgb_array'))

[env.step(0) for x in range(10000)]
plt.figure()
plt.imshow(env.render(mode='rgb_array'))

env.render(close=True)
