import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
    sys.path.append("../")

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

# create the learning environment of mountain car
env = gym.envs.make("MountainCar-v0")

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
# Standardize features by removing the mean and scaling to unit variance
scaler = sklearn.preprocessing.StandardScaler()

# Compute the mean and std to be used for later scaling.
scaler.fit(observation_examples)

# Used to convert a state to a featurized representation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

# linear approximation function
class Estimator():
    """
    Value function approximator.
    """
    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            # Linear model fitted by minimizing a regularized empirical loss with SGD
            model = SGDRegressor(learning_rate="constant", tol=0.21)
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            model.partial_fit([self.featurize_state(env.reset())],[0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Make value function predictions.

        :param s: state to make a prediction for
        :param a: (Optional) action to make a prediction for
        :return:
        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    :param env: OpenAI environment
    :param estimator: Action-Value function estimator
    :param num_episodes: Number of episodes to run for.
    :param discount_factor: Lambda time discount factor.
    :param epsilon: Chance the sample a random action. Float between 0 and 1.
    :param epsilon_decay: Each episode, epsilon is decayed by this factor.
    :return: An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        # The policy we're following with episode decay
        policy = make_epsilon_greedy_policy(
                estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # Only used for SARSA, not Q-learning
        next_action = None

        # One step in the environment
        for t in itertools.count():
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action

            # Take a step
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD update
            q_value_next = estimator.predict(next_state)

            # Use this code for Q-learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_value_next)

            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            # td_target = reward + discount_factor * q_values_next[next_action]

            # Update the function approximator using our target
            estimator.update(state, action, td_target)

            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")

            if done:
                break

            state = next_state

    return stats

estimator = Estimator()

# Note: For the Mountain Car we don't actually need an epsilon > 0.0
# because our initial estimate for all states is too "optimistic" which leads
# to the exploration of all states.
stats = q_learning(env, estimator, 100, epsilon=0.0)

plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)