import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
    sys.path.append("../")

from collections import defaultdict
from lib.env.windy_gridworld import WindyGridworldEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = WindyGridworldEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    :param Q: A dictionary that maps from state -> action-values
            Each value is a numpy array of length nA
    :param epsilon: The probability to select a random action. float between 0 and 1
    :param nA: Number of actions in the environment

    :return: A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    :param env: OpenAI environment.
    :param num_episodes: Number of episodes to run for.
    :param discount_factor: Lambda time discount factor
    :param alpha: TD learning rate.
    :param epsilon: Chance the sample a random action. Float between 0 and 1
    :return: A tuple (Q, episode_lengths).
    """
    # The final action-value function.

    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()

        for t in itertools.count():
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha *td_delta

            if done:
                break;

            state = next_state

    return Q, stats

Q, stats = q_learning(env, 500)
plotting.plot_episode_stats(stats)