
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random 
import os 
import sys
import time 

from collections import deque 
from env import SimpleAmpEnv
from DQNagent import DQNagent


# Initialize environment 
env = SimpleAmpEnv(ideal=False, reward_type="AutoCkt")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n 

# Check the input layer size, and Output layer size 
print(f"Input Layer size: {state_size}")
print(f"Output Layer size: {action_size}")

# Initialize agent
agent = DQNagent(state_size, action_size)

scores = []
num_episodes = 100

# Iteration

for episode in range(num_episodes):

    done = False 
    step = 0
    score = []
    state = np.reshape(env.reset(), [1, state_size])
    dummy = 0
    lt2 = time.time() 
    est = time.time()
    total_lt = []


    while not done: 
        step += 1
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action) 
        next_state = np.reshape(next_state, [1, state_size])
        agent.append_sample(state, action, reward, next_state, done)
        score.append(reward) 

        if (len(agent.memory) >= agent.train_start) & (step % 32 == 0)  :
            lt1 = time.time()
            agent.train_model()
            lt2 = time.time() 
            total_lt.append(lt2-lt1)
            print(f"step: {step} , time elapsed : {sum(total_lt)}", end='\r')

        if (step+1) % 10 == 0 :
            agent.update_target_model()

        if done:
            result = np.average(score)
            scores.append(result) 
            print(f"Episode {episode+1} =>", end=" ")
    print(f"total steps: {step}")
    print(f"learning time / episode : {np.sum(total_lt):.2f}")
    print(f"total elapsed time: {time.time()-est:2f}")

plt.figure(figsize=(9, 5))
plt.plot(scores, label="mean episode reward")
plt.ylim([-1, 1])
plt.legend()
plt.grid() 