# Import Packages that will be used in the agent

import sys
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np

from collections import deque
from tensorflow.keras.layers import Dense 
from tensorflow.keras.losses import mae, mse, Huber
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.models import Sequential 


class DQNagent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        self.model_sum_verbose = False

        # Initialize the state_size and action_size
        self.state_size = state_size
        self.action_size = action_size

        # Initialize DQN hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.batch_size = 32
        self.train_start = 1000

        # Initialize Memory of size 2000
        self.memory = deque(maxlen=50000)

        # Make main_model and Target_model
        self.main_model = self.build_model()
        self.target_model = self.build_model()

        # Initialize target_model
        self.update_target_model()

        # load model weights
        if self.load_model:
            self.main_model.load_weights("./save_model/dqn_trained.h5")
        
    
    # build model
    def build_model(self):
        # Set random seed for reproducibility
        tf.random.set_seed(42)

        # 1) Build model
        model = Sequential()
        model.add(Dense(5, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))

        # 2) Compile the model
        model.compile(loss=mse, optimizer=Adam(learning_rate=self.learning_rate), metrics=['mse'])

        # 3) Summarize the model
        if self.model_sum_verbose:
            model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())
    
    # select action by epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.main_model.predict(state)
            return np.argmax(q_value[0])
    
    # load sample in the replay buffer
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    # train model with random replay buffer sample
    def train_model(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # random extractions from memory with the size of batch size
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])
        
        # Q-value of the main_model of current state
        target = self.main_model.predict(states)
        
        # Q-value of the target_model of next states
        target_value = self.target_model.predict(next_states)

        # Update targets using Bellman Equation
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_value[i]))
        
        self.main_model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)



if __name__ == "__main__":
    
    num_episode=500
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n 

    # Initialize a DQN agent
    agent = DQNagent(state_size, action_size)

    scores, episodes = [], []

    for e in range(num_episode):
        done = False
        score = 0
        # Initialize Environment
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render():
                env.render()
            
            # decided an action based on current state
            action = agent.get_action(state)
            # progress a step with current action
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # give -100 reward if episode ends during progress
            reward = reward if not done or score==499 else -100

            # load <s, a, r, s'> in replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # learn every time step
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
            
            score += reward
            state = next_state

            if done:
                # update target models weight with main_models weight
                agent.update_target_model()

                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                plt.pyplot(episodes, scores, 'b')
                print("Episode:", e, " score: ", score, " memory length: ", len(agent.memory), " epsilon:", agent.epsilon)

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.model.save_weights("./save_model/dqn_trained.h5")
                    sys.exit()






