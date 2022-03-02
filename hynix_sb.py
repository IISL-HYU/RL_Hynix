
# test stable baselines 
import gym 

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make("CartPole-v1")

model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=2)

mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=1000)

print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


