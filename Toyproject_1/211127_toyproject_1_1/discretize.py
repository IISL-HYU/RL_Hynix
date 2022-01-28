import gym
import numpy as np

from gym.spaces import Discrete, Box
from proto import SimpleAmpEnv

# class DiscretizedObservationWrapper(gym.ObservationWrapper):

#     """
#     This class makes and environment with Box spaces to an environment with Discrete spaces
#     """
#     def __init__(self, env, n_bin=10, low=None, high=None):
#         super.__init__(env)
#         assert isinstance(env.observation_space, Box)

#     def _discretize_observation(self,)
    
#     def observation(self, observation):

#         return observation

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """This wrapper converts a Box observation into a single integer.
    """
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.observation_space = Discrete(n_bins ** low.flatten().shape[0])

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)



if __name__ == "__main__":
    env = SimpleAmpEnv()
    # pre_obs = {}
    # pre_obs["observation space"] = env.observation_space
    # pre_obs["Lower bound"] = env.observation_space.low
    # pre_obs["upper bound"] = env.observation_space.high
    # print(pre_obs)

    wrapped_env = DiscretizedObservationWrapper(
        env,
        n_bins=100,
    )

    post_obs = {}
    post_obs["observation space"] = wrapped_env.observation_space
    post_obs["Size of space"] = wrapped_env.observation_space.n
    print(post_obs)

    obs = wrapped_env.observation_space.sample()
    print(obs)

