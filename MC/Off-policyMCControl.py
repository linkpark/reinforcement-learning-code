import gym
import matplotlib
import numpy as np
import sys

from  collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")

from lib.env.blackjack_env import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()


# why we need off-policy reinforcement learning?
# A key benefit of this separation is that the behavior policy can operate by sampling all actions, whereas the estimation policy can be deterministic (e.g., greedy)

def create_random_policy(nA):
    """
    Creates a random policy function
    :param nA: Number of actions in the environment
    :return: A function that takes an observation as input and returns a vector
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):

        return A

    return policy_fn

def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values
    :param Q: A dictionary that maps from state -> action values
    :return: A function that takes an observation as input and returns a vector of action probabilities
    """

    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A

    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Mote Carlo Control Off-Policy Control using Weighted Importance Sampling
    :param env: OpenAI gym environment
    :param num_episodes: number of episodes to sample
    :param behavior_policy: the behavior policy used to generate episodes
    :param discount_factor: Lambada discount factor
    :return:A tuple (Q, policy), Q is state action function, policy is a function mapping from state to action
    """

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    # Our greedily policy to learn
    target_policy = create_greedy_policy(Q)

    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print("\rEpisode{}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode
        # An episode is an array of (state, action, reward) tuples
        episodes = []
        state = env.reset()
        for t in range(100):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episodes.append((state, action, reward))

            if done:
                break;
            state = next_state

        # Sum of discounted returns
        G = 0.0

        # The importance sampling ratio (the weights of the returns)
        W = 1.0

        # For each step in the episode, backwards( For t = T-1, T-2, ... down to 0)
        for t in range(len(episodes))[::-1]:
            state, action, reward = episodes[t]
            # Update the total reward since step t
            G = discount_factor * G + reward
            # update weighted importance sampling formula denominator
            C[state][action] += W

            # Update the action-value function using the incremental update formula
            # This also improves our target policy the probability will be 0 and we can break
            Q[state][action] += (W / C[state][action]) * ( G - Q[state][action])

            if action != np.argmax(target_policy(state)):
                break

            W = W * (1. / behavior_policy(state)[action])

    return Q, target_policy

random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")




