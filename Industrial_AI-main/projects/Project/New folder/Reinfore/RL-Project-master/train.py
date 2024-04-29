import gym_examples
import gymnasium as gym
import time




env = gym.make('gym_examples/CrowdNav-v0')
env.action_space.seed(42)

episode = 20000
episode_num = 1

observation, info = env.reset(seed=episode_num)

while True:

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if (terminated or truncated):
        episode_num += 1
        if (episode_num <= episode):
            observation, info = env.reset(seed=episode_num)
        else:
            break

env.close()
