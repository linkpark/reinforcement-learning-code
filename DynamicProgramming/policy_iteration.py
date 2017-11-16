import numpy as np
import pprint
import sys

if "../" not in sys.path:
    sys.path.append("../")

from lib.env.grid_world_env import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    V = np.zeros(env.nS)
    while True:
        delta = 0
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

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        old_policy = policy
        V = policy_eval_fn( policy, env, discount_factor)

        policy_stable = True

        for s in range(env.nS):
            chosen_action = np.argmax(policy[s])
            action_values = np.zeros(env.nA)

            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    # using += to implement accumulation \sum
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
                best_action = np.argmax(action_values)

            if chosen_action != best_action:
                policy_stable = False


            # np.eye return a 2-D array with ones on the diagonal and zeros elsewhere.
            policy[s] = np.eye(env.nA)[best_action]

        if policy_stable:
            return policy, V


# test policy iteration

policy, v = policy_improvement(env)
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


# policy evaluation and policy improvement loop is the basic framework for reinforcement learning algorithms