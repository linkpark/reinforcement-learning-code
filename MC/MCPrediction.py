import gym
import matplotlib
import numpy as np
import sys


# collections library is a high-performance container data types
from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")

from lib.env.blackjack_env import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def mc_prediction(policy, env, num_episodes, discount_factor = 1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function for a given
    policy using sampling.

    :param policy: A function that maps an observation to action probabilities
    :param env: OpenAI gym environment
    :param num_episodes: Number of episodes to sample
    :param discount_factor: Lambda discount factor
    :return: A dictionary that maps from state -> value.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode{}/{}.".format(i_episode, num_episodes, end=""))
            sys.stdout.flush()

        # Generate an episode
        # An episode is an array of (state, action, reward) tuples
        episodes = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episodes.append((state, action, reward))

            if done:
                break;
            state = next_state

        states_in_episode = set([tuple(x[0]) for x in episodes])

        for state in states_in_episode:
            first_occurence_idx = next(i for i, x in enumerate(episodes) if x[0] == state)

            G = sum([x[2]*(discount_factor ** i ) for i,x in enumerate(episodes[first_occurence_idx:])])

            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V

def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >=20 else 1


V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")