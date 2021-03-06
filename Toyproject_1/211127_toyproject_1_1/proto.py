import gym
import numpy as np
import matplotlib.pyplot as plt

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

    BANDWIDTH = 50 * 1e9    # 50Grad/s
    Amp = 6
    VOLTAGE = 300 * 1e-3    # 300mV
    CAP_L = 100 * 1e-15     # 100fF


    def __init__(self, gradient=0.01, verbose=0, ideal=True):

        super(SimpleAmpEnv, self).__init__()
        self.observation_space = spaces.Box(low=np.array([0.0]),
                                            high=np.array([1.0]),
                                            shape=(1,),
                                            dtype=np.float32)
        self.gradient = gradient
        self.epsilon = 2 * gradient
        self.action_space = spaces.Discrete(3)
        self.current_id = 0.0
        self.gm = 0.0
        self.rd = 0.0
        self.gain_bw = 0.0
        self.verbose = verbose
        self.ideal = ideal
        self.time_step = 0
        self.optimal_time_step = 0


    def reset(self):
        self.time_step = 0
        # optimal 에서 반복 시 횟수 count하여 종료 variable
        self.optimal_time_step = 0
        self.current_id = np.random.random_sample()
        self.gm, self.rd, self.gain_bw = self._circuit_topology(current_id=self.current_id)
        return np.array([self.current_id]).astype(np.float32)

    
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
        
        if self.time_step >= 10 and self.optimal_time_step == 0:
            done = True
            reward = 0
        
        # Observation (state) will be changed after action
        if action == 2:
            self.current_id += self.gradient
        elif action == 0:
            self.current_id -= self.gradient
        elif action == 1:
            pass
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
 
        self.current_id = np.clip(self.current_id, self.gradient, 1)
        self.gm, self.rd, self.gain_bw = self._circuit_topology(current_id=self.current_id)        

        ### needs a little modification of rewards 
        if self.current_id < -0.01 or self.current_id >= 1.0:
            reward = -100
            done = True
        else:
            if self.gain_bw >= (self.Amp * self.BANDWIDTH):
                reward = 0
                if abs(self.gain_bw - (self.BANDWIDTH * self.Amp)) < (self.epsilon * 5e12):
                    self.optimal_time_step += 1
                    reward = 2
                    if self.optimal_time_step > 30:
                        done = True
            else:
                reward = -10

        return np.array([self.current_id]).astype(np.float32), reward, done, info


    def _circuit_topology(self, current_id):
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




        if self.verbose == 1:
            
            print("_____________________________________________________________")
            print(f"|gm computed by current action is {gm:.3f} s                    |")
            print(f"|rd computed by current action is {rd:.3f} Ohm                  |")
            print(f"|gain bandwidth by current ID is {(gain_bw)*1e-9:.1f}Grad/s                |")
            print("_____________________________________________________________")

        return gm, rd, gain_bw


if __name__ == "__main__":
    env = SimpleAmpEnv(gradient=0.001 ,verbose=1, ideal=False)
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

    action = 0
    obs = env.reset()
    obs[0] = 0.1
    done = False

    while not done:
        obs, reward, done, info = env.step(action)
        print(f'{obs[0]:.4f} A ', end=', ')
        # print(obs[0], reward, done, info)
        print("reward:", reward)



    


