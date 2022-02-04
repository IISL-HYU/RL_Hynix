import gym
import numpy as np
import matplotlib.pyplot as plt
import time

from gym import spaces
from numpy.core.fromnumeric import shape
from numpy.core.numeric import Inf
from numpy.lib.function_base import gradient

from stable_baselines3.common.env_checker import check_env

class SimpleAmpEnv(gym.Env):
    """
    Simple Amplifier Environment : may be a simple Linear regression program
    Custom Environment : follows gym environment
    :param gradient: (float) gradient of ID per step
    :param verbose:  (int) verbosity - 0: no output 1: information
    :param ideal: (boolean) is the circuit ideal?
    """
    INCREASE = 2    
    STABLE = 1
    DECREASE = 0
    REWARD_SET = {'inverse', 'decrease_inverse', 'decrease_one'}

    # This target bandwidth should be a parameter
    BANDWIDTH = 50 * 1e9    # 50Grad/s
    Amp = 6
    VOLTAGE = 300 * 1e-3    # 300mV
    CAP_L = 100 * 1e-15     # 100fF


    def __init__(self, gradient=0.001, verbose=0, ideal=True, reward_type='inverse'):

        super(SimpleAmpEnv, self).__init__()
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]),
                                            high=np.array([1.0, Inf, 1.0]),
                                            shape=(3,),
                                            dtype=np.float32)


        self.gradient = gradient
        self.action_space = spaces.Discrete(3)

        # Initialize Currents
        self.current_id = 0.0
        self.previous_id = 0.0
        self.gm = 0.0
        self.rd = 0.0
        self.gain_bw = 0.0
        self.observation_bounds = np.array([.3, 600e9, 600e9])
        self.verbose = verbose
        self.ideal = ideal
        self.time_step = 0
        self.reward_type = reward_type


    def reset(self):
        self.time_step = 0
        self.current_id = np.random.randint(300) * 1e-3     # random sampling between 0.0~0.3
        self.gm, self.rd, self.gain_bw = self._circuit_topology(current_id=self.current_id)
        obs = np.array([self.current_id, self.gain_bw, (self.Amp * self.BANDWIDTH)]).astype(np.float32)
        obs = self.normalize_target(obs)
        return obs

    
    def step(self, action):
        """
        Works as a circuit
        :param action: (gradient of ID)
        :return (np.ndarray, float, boolean, dict): observation, reward, is the episode done, information
        """
        reward = 0 
        info = {}
        done = False
        self.time_step += 1
        self.previous_id = self.current_id

        if self.time_step >= 1000:
            done = True
            reward = 0

        # Observation (state) might change after action
        if action == 2:
            self.current_id += self.gradient
        elif action == 0:
            self.current_id -= self.gradient
        elif action == 1:
            pass
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        self.gm, self.rd, self.gain_bw = self._circuit_topology(current_id=self.current_id)        
 
        reward = self.get_reward(self.reward_type)
        
        if self.current_id <= 0.0 or self.current_id >= 1.0:

            # for checking the environment
            if __name__ == '__main__':
                done = True
            reward = -100
            self.current_id = self.previous_id
        obs = np.array([self.current_id, self.gain_bw, (self.Amp*self.BANDWIDTH)]).astype(np.float32)
        obs = self.normalize_target(obs)
        return obs, reward, done, info


    # normalized current observations via bounded max values
    def normalize_target(self, observations):
        return np.array(observations/self.observation_bounds).astype(np.float32)
        
    # return reward by case of individual reward types
    def get_reward( self, reward_type):
        reward = 0
        
        if self.gain_bw >= (self.Amp * self.BANDWIDTH):
            
            if reward_type == "inverse":
                reward = (1 / self.previous_id)           
            elif reward_type == "decrease_inverse":
                if self.previous_id > self.current_id:
                    reward += (1 / self.previous_id)
                elif self.previous_id == self.current_id:
                    pass
                else:
                    reward =-20
            elif reward_type == "decrease_one":
                if self.previous_id >= self.current_id:
                    reward += 1
            else:
                raise Exception("Incorrect keyword")
                            
        else:
            reward = -20
        return reward

    # returns every computed values of the topology
    def _circuit_topology(self, current_id):
        # For Advanced works, the _circuit_topology function might be replaced to a function
        # that outputs an circuit input dictionary to a simulator
        # and gets the simulated values as a dictionary format 
        # and returns the simulated values 
        """
        Circuit topology : computes Resistance, Gain Bandwidth, with respect to current id 
        :param current_id: current id 
        :return (float), (float), (float) : gm, R_D, Gain_Bandwidth
        """

        gm = (2 * current_id) / self.VOLTAGE  # siemens
        rd = self.Amp / gm              # Ohm

        if self.ideal:
            gain_bw = self.Amp / (rd * self.CAP_L)
        else:
            cap_d = ((1e-15)/(100 * 1e-6))*current_id
            gain_bw = self.Amp / (rd * (self.CAP_L + cap_d))

        # If the User selects verbosity = 1, the environment prints out the computed values
        if self.verbose == 1:
            
            print("_____________________________________________________________")
            print(f"Current {current_id}")
            print(f"|gm computed by current action is {gm:.3f} s                    |")
            print(f"|rd computed by current action is {rd:.3f} Ohm                  |")
            print(f"|gain bandwidth by current ID is {(gain_bw)*1e-9:.1f}Grad/s                |")
            print("_____________________________________________________________")

        return gm, rd, gain_bw


if __name__ == "__main__":
    env = SimpleAmpEnv(gradient=0.001 ,verbose=1, ideal=False)
    # print(env.reset())
    check_env(env, warn=True)

    # print(f"initial I_D : {obs}")
    # print(f"Target gain Bandwidth = {env.BANDWIDTH}")

    # print()
    # for action in range(3):
    #     obs = env.reset()

    #     print("current action: ", action)
    #     print(f"Initial ID ={obs[0]:.3f}A")

    #     obs, reward, done, info = env.step(action)
    #     print(f"ID after action {action}: {obs[0]:.3f}A", end='\n\n')

    # action = 0
    # obs = env.reset()
    # obs[0] = 0.1
    # done = False

    # while not done:
    #     obs, reward, done, info = env.step(action)
    #     print(f'{obs[0]:.4f} A , {obs[1]:.3f}, {obs[2]:.3f}', end=', ')
    #     # print(obs[0], reward, done, info)
    #     print("reward:", reward)



    


