import gym_examples_v2
import gymnasium as gym
import time

env = gym.make('gym_examples_v2/CrowdNav-v0', render_mode='human')
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
