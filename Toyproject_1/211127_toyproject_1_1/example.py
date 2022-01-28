import gym
import numpy as np
import matplotlib.pyplot as plt

from gym.spaces import Discrete, Box
from proto import SimpleAmpEnv
from discretize import DiscretizedObservationWrapper as DO
from collections import defaultdict



env = SimpleAmpEnv(gradient=0.001, ideal=False)
env = DO(env, 1000)
actions = range(env.action_space.n)

Q = defaultdict(float)
print(Q)
gamma = 0.99
alpha = 0.2
epsilon = 0.01

def update_Q(s, r, a, next_state, done):
    max_q_next = max([Q[next_state, a] for a in actions]) 
    # Do not include the next state's value if currently at the terminal state.
    Q[s, a] =  (1 - alpha) * Q[s, a] + alpha * (r + gamma * max_q_next)

def act(obs):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    if obs > 20:
        return 0

    qvals = {a: Q[obs, a] for a in actions}
    max_q = max(qvals.values())
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)


obs = env.reset()
rewards = []
reward = 0.0
print("observation = ", obs)
action = act(obs)
print("action: ", action)

for step in range(2000000):
    action = act(obs)
    next_obs, re, done, info = env.step(action)
    # print(next_obs, end=',')
    update_Q(obs, re, action, next_obs, done)
    reward += re
    if done:
        rewards.append(reward)
        reward = 0.0
        obs = env.reset()
        # print()
    else:
        obs = next_obs


for j in range(100):
    print("state ",j, ":",np.argmax([Q[j, i] for i in actions]))

for i in range(1000):
    if i < 30 or i % 20 == 0 :
        print(i, ':', Q[i, 0], Q[i, 1], Q[i, 2])

# plt.figure(1)
# plt.plot(rewards)
# plt.show()

