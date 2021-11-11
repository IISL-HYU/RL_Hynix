# Make custom environment

import gym
import numpy as np
from gym import spaces
from numpy.lib.function_base import gradient


# Custom Environment build-up

class HorizontalMazeEnv(gym.Env):
    """
    Custom Environment : follows gym environment
    """
    INCREASE = 1
    DECREASE = 0
    # TARGET = np.random.random_sample() * 2 - 1

    # Target value is given (but the agent doesn't know)
    TARGET = 0.65


    def __init__(self, gradient=0.1, eps=0.2):
        """
        1-dimensional Maze environment: which might suit simple linear problems
        :param gradient: (float) gradient that will change the state
        :param eps: (float) stopping criterion
        """
        super(HorizontalMazeEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.gradient = gradient
        self.currest_pos = 0.0
        self.eps = eps
        self.init_pos = 0.0
        

    def reset(self):
        # Current Position will be reset (randomly)
        self.current_pos = np.random.random_sample() * 2 - 1
        self.init_pos = self.current_pos
        # self.TARGET = np.random.random_sample() * 2 - 1
        return np.array([self.current_pos]).astype(np.float32)

    def step(self, action):
        """
        :param action: (int) Action that the agent will take
        :return (np.ndarray, int, boolean, dict): observation, reward, done, information
        """
        reward = 0
        info = {}


        # Updates the current observation
        if action == 1:
            self.current_pos += self.gradient
        elif action == 0:
            self.current_pos -= self.gradient
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        # make sure that the current position sticks inside the interval
        self.current_pos = np.clip(self.current_pos, -1.0, 1.0)

        # Stopping criterion and returns reward and boolean
        (reward, done) = (-1, False) if (abs(self.current_pos - self.TARGET) >= self.eps) else (20, True)

        # Information (Target Value, Initial position)
        info['Target Value'] = self.TARGET
        info['Initial position'] = self.init_pos



        return np.array([self.current_pos]).astype(np.float32), reward, done, info




if __name__ == '__main__':
    
    # toy test for gym
    toy_env = gym.make("CartPole-v1")
    obs = toy_env.reset()
    reward_episode = []
    for _ in range(10):
        action = toy_env.action_space.sample()
        obs, reward, done, info = toy_env.step(action=action)
        reward_episode.append(reward)
    print(np.sum(reward_episode))

    # Test our gym environment
    env = HorizontalMazeEnv(gradient=0.01, eps=0.02)
    print(env.observation_space)
    print(env.observation_space.sample())
    obs = env.reset()
    for step in range(100):
        action = env.action_space.sample()
        print("Action at step {}: {}".format(step+1, action))
        obs, reward, done, info = env.step(action)
        print("obs =", obs, "reward =", reward, "done =", done)
        print()
        if done == True:
            break

    print("obs =", obs, "reward =", reward, "done =", done, "info =", info)
