import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from lib.env.grid_world_env import GridworldEnv

env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            v = 0
            # extract action & action probability from policy
            # enumerate returns a list's index and its value
            for a, action_prob in enumerate(policy[s]):
                for transition_prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * transition_prob * (reward + discount_factor * V[next_state])

            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        if delta < theta:
            break

    return np.array(V)


# Test policy evaluation
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

print("Value Function:")
print(v)
