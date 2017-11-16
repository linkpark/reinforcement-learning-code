import numpy as np
import pprint
import sys

if "../" not in sys.path:
    sys.path.append("../")

from lib.env.grid_world_env import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor =1.0):
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        delta = 0.0
        for s in range(env.nS):
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for transition_prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += transition_prob * (reward + discount_factor * V[next_state])

            best_action = max(action_values)
            delta = max(delta, np.abs(action_values[best_action]- V[s]))

            V[s] = action_values[best_action]
            policy[s] = np.eye(env.nA)[best_action]

        if delta < theta:
            break

    return policy, V

#test value iteration
policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")