import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.envs.make("Breakout-v0")

print("Action space size: {}".format(env.action_space.n))

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

plt.figure()
plt.imshow(env.render(mode='rgb_array'))

[env.step(2) for x in range(1)]
plt.figure()
plt.imshow(env.render(mode='rgb_array'))

env.render(close=True)

# Check out what a cropped image looks like
plt.imshow(observation[34:-16,:,:])